class FileManager:
    def __init__(self):
        self.file_content = ""

    def load_md_file(self, file_path: str)-> str:
        """
        Carga el contenido de un archivo .md y lo asigna a la variable file_content.
        
        :param file_path: Ruta al archivo .md que se va a cargar.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.file_content = file.read()
            return self.file_content 
        except FileNotFoundError:
            print(f"Error: El archivo {file_path} no se encontró.")
        except IOError as e:
            print(f"Error al leer el archivo {file_path}: {e}")

    def get_content(self):
        """
        Devuelve el contenido del archivo cargado.
        
        :return: Contenido del archivo .md.
        """
        return self.file_content