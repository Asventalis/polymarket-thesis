from pathlib import Path

path = Path("src/polymarket_gnn/data/causal_dataset.py")
content = path.read_text()

# Replace hardcoded times in __init__
init_old = """    def __init__(
        self,
        date_str: str,"""
init_new = """    def __init__(
        self,
        date_str: str,
        start_hour: int = 0,
        end_hour: int = 23,"""
content = content.replace(init_old, init_new)

init_assign_old = """        self.date_str = date_str
        self.n_markets = n_markets"""
init_assign_new = """        self.date_str = date_str
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.n_markets = n_markets"""
content = content.replace(init_assign_old, init_assign_new)

# Replace start_time / end_time assignments in the methods
old_time1 = """        start_time = f"{self.date_str} 00:00:00"
        end_time = f"{self.date_str} 23:59:59\""""
new_time1 = """        start_time = f"{self.date_str} {self.start_hour:02d}:00:00"
        end_time = f"{self.date_str} {self.end_hour:02d}:59:59\""""
content = content.replace(old_time1, new_time1)

old_time2 = """        start_time = f"{self.date_str} 00:00:00"
        end_time   = f"{self.date_str} 23:59:59\""""
new_time2 = """        start_time = f"{self.date_str} {self.start_hour:02d}:00:00"
        end_time   = f"{self.date_str} {self.end_hour:02d}:59:59\""""
content = content.replace(old_time2, new_time2)

path.write_text(content)
print("Patched causal_dataset.py for hourly windows")
