const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:5173');

  // Find file input and upload file
  await page.setInputFiles('input[type="file"]', '/home/bhumi/GitHub/NeuroAegis/data/chbmit_subset/chb01/chb01_01.edf');
  
  // Wait for some prediction or submit button to appear/become active and click it
  // (Assuming it auto-submits or there is an analyze button)
  try {
      await page.waitForSelector('button:has-text("Analyze")', { timeout: 3000 });
      await page.click('button:has-text("Analyze")');
  } catch (e) {
      console.log("No Analyze button found, maybe it auto-submits?");
  }
  
  // Wait for results
  await page.waitForTimeout(5000);
  
  await page.screenshot({ path: 'prediction_screenshot.png' });
  console.log("Screenshot saved to prediction_screenshot.png");
  
  // Navigate to model info tab if it exists
  try {
    await page.click('text="Reports"');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'reports_screenshot.png' });
  } catch (e) {}
  
  try {
    await page.click('text="Analysis"');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'analysis_screenshot.png' });
  } catch (e) {}

  await browser.close();
})();
