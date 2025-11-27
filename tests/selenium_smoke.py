# Standalone smoke test to verify Selenium Manager launches Chrome successfully.
# Run manually from an interactive session: `python test/selenium_smoke.py`

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def main() -> None:
    opts = Options()
    # 需要可视化观察，请勿开启无头模式
    opts.add_argument("--ozone-platform-hint=auto")
    # 如果以 root 身份运行才需要下面这一行，否则不要加
    # opts.add_argument("--no-sandbox")
    # WSL 的 /dev/shm 容量偏小，复杂页面建议加这一行
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=opts)  # Selenium Manager 将自动匹配 chromedriver
    driver.get("https://www.google.com")
    print(driver.title)
    # 观察交互完毕后再关闭
    # driver.quit()


if __name__ == "__main__":
    main()
