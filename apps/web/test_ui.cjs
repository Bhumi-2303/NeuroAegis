const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:5173');

  // Find file input and upload file
  await page.setInputFiles('input[type="file"]', '/home/bhumi/GitHub/NeuroAegis/data/chbmit_subset/chb01/chb01_01.edf');
  
  // Wait for some prediction or submit button to appear/become active and click it
  try {
      await page.waitForSelector('button:has-text("Analyze")', { timeout: 3000 });
      await page.click('button:has-text("Analyze")');
  } catch (e) {
      console.log("No Analyze button found, maybe it auto-submits?");
  }
  
  // Wait for results
  await page.waitForTimeout(6000);
  
  await page.screenshot({ path: '/home/bhumi/.gemini/antigravity/brain/becc8203-cd24-4c90-a01e-ff0d771d6241/artifacts/prediction_screenshot.png' });
  console.log("Screenshot saved to prediction_screenshot.png");
  
  // Navigate to model info tab if it exists
  try {
    await page.click('text="Reports"');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/home/bhumi/.gemini/antigravity/brain/becc8203-cd24-4c90-a01e-ff0d771d6241/artifacts/reports_screenshot.png' });
  } catch (e) {}
  
  try {
    await page.click('text="Analysis"');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/home/bhumi/.gemini/antigravity/brain/becc8203-cd24-4c90-a01e-ff0d771d6241/artifacts/analysis_screenshot.png' });
  } catch (e) {}
  
  try {
    await page.click('text="SHAP"');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/home/bhumi/.gemini/antigravity/brain/becc8203-cd24-4c90-a01e-ff0d771d6241/artifacts/shap_screenshot.png' });
  } catch (e) {}

  await browser.close();
})();
