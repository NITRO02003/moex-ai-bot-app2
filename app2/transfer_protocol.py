# app2/transfer_protocol.py
from datetime import datetime
from pathlib import Path
import json
from .paths import REPORTS_DIR

TRANSFER_DIR = REPORTS_DIR / 'transfer'
TRANSFER_DIR.mkdir(exist_ok=True)

AUTO_REPORT_PATH = TRANSFER_DIR / 'auto_report.md'
MANUAL_SUMMARY_PATH = TRANSFER_DIR / 'manual_summary.md'


class TransferProtocol:
    def __init__(self):
        self.auto_report_path = AUTO_REPORT_PATH
        self.manual_summary_path = MANUAL_SUMMARY_PATH

    def update_auto_report(self,
                           latest_results: dict = None,
                           current_experiments: str = None,
                           problems: str = None,
                           decisions: str = None):
        """Обновляет автоматический отчет"""

        content = f"""# MOEX-AI-BOT AUTOMATED REPORT
**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## CURRENT STATUS
{self._get_current_status(latest_results)}

## CURRENT EXPERIMENTS
{current_experiments or "No active experiments recorded"}

## ACTIVE PROBLEMS  
{problems or "No active problems recorded"}

## RECENT DECISIONS
{decisions or "No recent decisions recorded"}

## ITERATION HISTORY
{self._get_iteration_history()}

---
*This report is auto-generated. Update via `transfer_protocol.py`*
"""

        with open(self.auto_report_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def create_manual_summary(self,
                              project_stage: str,
                              critical_decisions: str,
                              next_priorities: str,
                              yandex_disk_link: str = None):
        """Создает ручной шаблон для переноса в новый чат"""

        content = f"""# MOEX-AI-BOT MANUAL TRANSFER SUMMARY
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## PROJECT STAGE
{project_stage}

## CRITICAL DECISIONS TO PRESERVE
{critical_decisions}

## NEXT PRIORITIES  
{next_priorities}

## AUTOMATED REPORT LINK
{yandex_disk_link or "Not uploaded yet"}

## INSTRUCTIONS FOR NEXT CHAT
1. Download automated report from Yandex.Disk
2. Read this manual summary first
3. Continue from the last experiment
4. Preserve the decision history

---
*Create this summary when reaching 95% token limit*
"""

        with open(self.manual_summary_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _get_current_status(self, latest_results: dict) -> str:
        if not latest_results:
            return "No recent results"

        status = []
        for symbol, metrics in latest_results.items():
            status.append(f"- **{symbol}**: {metrics.get('total_trades', 0)} trades, "
                          f"WR: {metrics.get('win_rate', 0):.1%}, "
                          f"Return: {metrics.get('total_return', 0):.2%}")

        return "\n".join(status)

    def _get_iteration_history(self) -> str:
        # Можно читать из файла историю, пока заглушка
        return """
- 2025-11-11: First regime strategy attempt - 0 trades
- 2025-11-11: Lowered thresholds in dynamic_regime_detector
- 2025-11-11: Added data quality diagnostics
"""