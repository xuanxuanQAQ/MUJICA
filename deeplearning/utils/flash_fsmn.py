import torch
from torch import nn
import torch.nn.functional as F
import copy

from ..utils.Transformer import TransformerEncoder_FLASH_DualA_FSMN

class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x

def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)
    

class Encoder(nn.Module):
    """Convolutional 2D Encoder Layer.

    Arguments
    ---------
    kernel_size : int or tuple
        Size of the convolving kernel. If int, same size for both dimensions.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple, optional
        Stride of the convolution. Default: kernel_size // 2

    """

    def __init__(self, kernel_size=3, in_channels=1, out_channels=64):
        super(Encoder, self).__init__()
        
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            # Use kernel_size // 2 as stride for int input
            self.stride = (kernel_size // 2, kernel_size // 2)
        else:
            self.kernel_size = kernel_size
            # Use kernel_size // 2 as stride for tuple input
            self.stride = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Ensure stride is at least 1
        self.stride = (max(1, self.stride[0]), max(1, self.stride[1]))
            
        # Remove padding - set to 0
        self.padding = (0, 0)
            
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels
        
        # Store original input size for decoder reconstruction
        self.original_size = None

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, H, W] or [B, C, H, W].
            
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, H', W'].
        original_size : tuple
            Original spatial dimensions (H, W) for decoder reconstruction.
        """
        # Store original spatial size
        if x.dim() == 3:
            self.original_size = (x.size(1), x.size(2))  # (H, W)
        elif x.dim() == 4:
            self.original_size = (x.size(2), x.size(3))  # (H, W)
        
        # Handle different input dimensions
        if x.dim() == 3:  # B x H x W -> B x 1 x H x W
            if self.in_channels == 1:
                x = torch.unsqueeze(x, dim=1)
            else:
                raise ValueError(f"Expected {self.in_channels} channels, got 1 channel input")
        elif x.dim() == 4:  # B x C x H x W
            if x.size(1) != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)} channels")
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D input")
        
        # B x C x H x W -> B x N x H' x W' (spatial dimensions reduced by stride)
        x = self.conv2d(x)
        x = F.relu(x)

        return x, self.original_size


class Decoder(nn.Module):
    """A 2D decoder layer that consists of ConvTranspose2d.
    
    This decoder reverses the operation of the corresponding Encoder,
    upsampling the spatial dimensions and changing the number of channels.
    Guarantees output shape matches the original input shape to encoder.

    Arguments
    ---------
    kernel_size : int or tuple
        Size of the convolving kernel. If int, same size for both dimensions.
        Should match the encoder's kernel_size.
    in_channels : int
        Number of input channels (should match encoder's out_channels).
    out_channels : int
        Number of output channels (should match encoder's in_channels).

    """

    def __init__(self, kernel_size=3, in_channels=64, out_channels=1, **kwargs):
        super(Decoder, self).__init__()
        
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            # Use kernel_size // 2 as stride to match encoder
            self.stride = (kernel_size // 2, kernel_size // 2)
        else:
            self.kernel_size = kernel_size
            self.stride = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Handle stride=0 case (when kernel_size < 2)
        self.stride = (max(1, self.stride[0]), max(1, self.stride[1]))
        
        # No padding to match encoder behavior
        self.padding = (0, 0)
        
        # Calculate output_padding to achieve proper upsampling
        self.output_padding = (self.stride[0] - 1, self.stride[1] - 1)
        
        # Initialize ConvTranspose2d with appropriate parameters
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=kwargs.get('bias', False),
        )
        
        self.out_channels = out_channels

    def forward(self, x, target_size=None):
        """Return the decoded output with exact target size.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, H', W'] or [B, H', W'].
        target_size : tuple, optional
            Target spatial size (H, W). If provided, output will be resized to match exactly.

        Return
        ------
        x : torch.Tensor
            Decoded tensor with exact target dimensions.
        """
        
        # Ensure input is 4D [B, C, H, W]
        if x.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            x = torch.unsqueeze(x, 1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D input")
        
        # Apply transposed convolution
        x = self.conv_transpose(x)
        
        # Apply activation (matching encoder's ReLU)
        x = F.relu(x)
        
        # Resize to exact target size if provided
        if target_size is not None:
            current_size = (x.size(2), x.size(3))
            if current_size != target_size:
                # Use interpolation to get exact size
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # If output has single channel, squeeze the channel dimension
        if self.out_channels == 1:
            x = torch.squeeze(x, dim=1)  # [B, 1, H, W] -> [B, H, W]
        
        return x
    
class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000):
        super(RoPENd, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        # create a stack of angles multiplied by position
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)

        # store in a buffer so it can be saved in model parameters
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        pe_x = self.rotations * x
        return torch.view_as_real(pe_x).flatten(-2)

class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx

class Dual_Flash_FSMN_BLOCK_Wrapper(nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
    ):

        super(Dual_Flash_FSMN_BLOCK_Wrapper, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")


        self.mdl = TransformerEncoder_FLASH_DualA_FSMN(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        output = self.mdl(x)

        return output

class Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        intra_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

        # Linear
        if linear_layer_after_inter_intra:
            self.intra_linear = Linear(
                    out_channels, input_size=out_channels
            )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, S = x.shape
        # intra RNN
        # [B, S, N]
        intra = x.permute(0, 2, 1).contiguous() #.view(B, S, N)

        intra = self.intra_mdl(intra)

        # [B, S, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, S]
        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out
        
 
class SigSepAttnNet(nn.Module):
    
    def __init__(self, in_channels=64, norm="ln", out_channels=64, 
                 num_layers=2, dropout=0.1, activation='relu',
                 intra_model=None,base=10000,
                skip_around_intra=True,
                linear_layer_after_inter_intra=True,):
        super(SigSepAttnNet, self).__init__()
        
        self.out_channels = out_channels
        self.base = base
        self.num_layers = num_layers
        self.in_channels = in_channels
        
        self.norm = select_norm(norm, in_channels, 4)
        self.conv2d = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        
        # 缓存不同尺寸的 RoPE
        self._rope_cache = {}
        
        self.mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.mdl.append(
                copy.deepcopy(
                    Computation_Block(
                        intra_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )
        
    def _get_rope(self, height, width):
        """获取或创建指定尺寸的 RoPE"""
        key = (height, width, self.out_channels)
        
        if key not in self._rope_cache:
            rope = RoPENd((height, width, self.out_channels), base=self.base)
            # 将 RoPE 移动到与模型相同的设备
            device = next(self.parameters()).device
            rope = rope.to(device)
            self._rope_cache[key] = rope
            
        return self._rope_cache[key]
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 任意尺寸
        Returns:
            x: [B, C, H, W] - 相同尺寸，应用了位置编码
        """
        B, C, H, W = x.shape
        
        x = self.norm(x)
        x = self.conv2d(x)
        
        # 获取对应尺寸的 RoPE
        rope = self._get_rope(H, W)
        
        # 应用位置编码
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = rope(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        x = x.view(B, C , -1)   # [B, H, W, C] -> [B, C, T]

        # 进行注意力机制计算
        for i in range(self.num_layers):
            x = self.mdl[i](x)
        x = self.prelu(x)

        x = self.output(x) * self.output_gate(x)

        # 展回
        x = x.view(B, C, H, W)  # 恢复原始形状        
        
        
        return x