from ui.main_window import MainWindow
from core.Data_fetcher import Data_fetcher
def main():
    Data_fetcher_instance = Data_fetcher("AAPL")
    info = Data_fetcher_instance.get_info()
    income = Data_fetcher_instance.get_and_edit_income_statement()
    print(info)
    print(income)
    server = MainWindow()
    server.run()

if __name__ == "__main__":
    main()
