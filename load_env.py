import os

def load_env(file_path=".env"):
    """
    Lê um arquivo .env e define as variáveis no ambiente do sistema.
    :param file_path: Caminho para o arquivo .env (padrão: .env)
    """
    if not os.path.exists(file_path):
        print(f"Arquivo {file_path} não encontrado.")
        return

    with open(file_path, "r") as env_file:
        for line in env_file:
            # Ignorar linhas vazias e comentários
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Separar chave e valor
            try:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
            except ValueError:
                print(f"Linha malformada no arquivo .env: {line}")

    print("Variáveis do arquivo .env carregadas no ambiente.")

# Exemplo de uso
if __name__ == "__main__":
    load_env()
