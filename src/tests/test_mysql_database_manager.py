from __future__ import annotations

from dotenv import load_dotenv

from agent.database import DatabaseConfig
from agent.mysql_database_manager import MysqlDatabaseManager


def main() -> None:
    """Quick manual test for MysqlDatabaseManager helpers."""
    load_dotenv()

    manager = MysqlDatabaseManager(DatabaseConfig.from_env())

    print("=== 所有表名 ===")
    tables = manager.list_tables()
    print(tables)

    print("\n=== 表与注释 ===")
    for item in manager.list_tables_with_comments():
        print(item)

    if not tables:
        print("\n数据库中没有任何表，无法继续测试。")
        return

    sample_table = tables[0]
    print(f"\n=== 表 {sample_table} 的结构 ===")
    for column in manager.describe_table(sample_table):
        print(column)

    sample_sql = f"SELECT * FROM {sample_table} LIMIT 5"
    print("\n=== SQL 验证 ===")
    print(manager.validate_sql(sample_sql))

    print("\n=== SQL 执行 ===")
    for row in manager.execute_sql(sample_sql):
        print(row)


if __name__ == "__main__":
    main()
