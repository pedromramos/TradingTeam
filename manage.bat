@echo off
set VENV_DIR=venv

REM Verificar se o Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo Python nao esta instalado ou nao foi adicionado ao PATH.
    pause
    exit /b 1
)

REM Criar o ambiente virtual se nao existir
if not exist %VENV_DIR% (
    echo Criando o ambiente virtual...
    python -m venv %VENV_DIR%
)

REM Ativar o ambiente virtual
if exist %VENV_DIR%\Scripts\activate (
    call %VENV_DIR%\Scripts\activate
) else (
    echo Falha ao ativar o ambiente virtual.
    pause
    exit /b 1
)

REM Instalar dependencias
if exist requirements.txt (
    echo Instalando dependencias...
    %VENV_DIR%\Scripts\pip install -r requirements.txt
) else (
    echo Arquivo requirements.txt nao encontrado.
    pause
    exit /b 1
)


REM Executar o script agents.py
if exist agents.py (
    echo Executando agents.py...
    %VENV_DIR%\Scripts\python agents.py
) else (
    echo Arquivo agents.py nao encontrado.
    pause
    exit /b 1
)

REM Manter o terminal aberto
pause
