param(
  [Parameter(Mandatory=$true)][string]$VenvPath,
  [Parameter(Mandatory=$true)][string]$Token,
  [Parameter(Mandatory=$true)][string]$AccountId
)

# Активируем нужное venv
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (!(Test-Path $activate)) {
  Write-Error "Не найден Activate.ps1 по пути: $activate"
  exit 1
}
. $activate

# Проверим версии и окружение
python -V
pip -V

# Ставим нужные пакеты (безопасно, если уже стоят — просто пропустит)
pip install -U pip
pip install tinkoff-investments pandas pyarrow fastparquet catboost joblib

# Пробрасываем секреты в окружение текущего процесса
$env:TINKOFF_TOKEN = $Token
$env:ACCOUNT_ID    = $AccountId

# Запускаем модуль загрузки данных (чтение 90 дней минуток по SBER,GAZP,LKOH)
python -m app.data_fetch
