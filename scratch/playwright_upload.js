const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://localhost:5173/');
  
  // The file input selector
  const fileInput = await page.$('input[type="file"]');
  await fileInput.setInputFiles('/home/bhumi/GitHub/NeuroAegis/data/chbmit_subset/chb01/chb01_01.edf');
  
  // Wait for the button to be enabled (Initialize Analysis)
  await page.waitForSelector('button:has-text("Initialize Analysis"):not([disabled])', { timeout: 10000 });
  await page.click('button:has-text("Initialize Analysis")');
  
  // Wait for the response or wait 15 seconds to let the backend process it
  await page.waitForTimeout(15000);
  
  console.log("Upload submitted and waited 15s.");
  
  await browser.close();
})();
