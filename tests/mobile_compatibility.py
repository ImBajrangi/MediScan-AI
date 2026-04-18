import asyncio
from playwright.async_api import async_playwright
import os

async def run_compatibility_test():
    async with async_playwright() as p:
        # Define devices to emulate
        devices = [
            {
                "name": "iPhone 14 (Safari)",
                "device": p.devices["iPhone 14 Pro"],
                "browser_type": p.webkit
            },
            {
                "name": "Pixel 7 (Chrome)",
                "device": p.devices["Pixel 7"],
                "browser_type": p.chromium
            }
        ]

        # Ensure results directory exists
        os.makedirs("evaluation_results/mobile_compatibility", exist_ok=True)

        for dev_config in devices:
            print(f"📡 Testing compatibility: {dev_config['name']}...")
            browser = await dev_config["browser_type"].launch()
            context = await browser.new_context(**dev_config["device"])
            page = await context.new_page()

            # Navigate to the dashboard (assumes app is running on 5001)
            try:
                await page.goto("http://localhost:5001/ckd", timeout=10000)
                
                # Check for key elements
                title = await page.title()
                is_visible = await page.is_visible("#clinical")
                
                print(f"  ✅ Page loaded: '{title}'")
                print(f"  ✅ Clinical module visible: {is_visible}")

                # Capture screenshot
                safe_name = dev_config["name"].replace(" ", "_").replace("(", "").replace(")", "").lower()
                screenshot_path = f"evaluation_results/mobile_compatibility/{safe_name}.png"
                await page.screenshot(path=screenshot_path)
                print(f"  📸 Screenshot saved: {screenshot_path}")

            except Exception as e:
                print(f"  ❌ Test failed for {dev_config['name']}: {e}")
            
            finally:
                await browser.close()

if __name__ == "__main__":
    asyncio.run(run_compatibility_test())
