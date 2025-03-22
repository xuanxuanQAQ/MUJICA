from TransFormat import binary_to_string


def data_save(data, file_name):
    with open(file_name, "w") as file:
            file.write(binary_to_string(data))