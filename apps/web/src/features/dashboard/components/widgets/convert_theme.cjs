const fs = require('fs');
const path = require('path');

const widgetsDir = '/home/bhumi/GitHub/NeuroAegis/apps/web/src/features/dashboard/components/widgets';

function processFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Text colors
    content = content.replace(/text-gray-[789]00/g, 'text-[var(--text-primary)]');
    content = content.replace(/text-gray-[56]00/g, 'text-[var(--text-secondary)]');
    content = content.replace(/text-gray-[34]00/g, 'text-[var(--text-muted)]');
    
    // Backgrounds
    content = content.replace(/bg-white/g, 'bg-transparent'); // Often not needed, or better transparent
    content = content.replace(/bg-gray-50\/[0-9]+/g, 'bg-[var(--bg-3)]');
    content = content.replace(/bg-gray-100\/[0-9]+/g, 'bg-[var(--bg-3)]');
    content = content.replace(/bg-gray-50(?!%)/g, 'bg-[var(--bg-3)]');
    content = content.replace(/bg-gray-100(?!%)/g, 'bg-[var(--bg-3)]');
    
    // Hover backgrounds
    content = content.replace(/hover:bg-gray-[15]00?/g, 'hover:bg-[var(--bg-4)]');
    
    // Borders
    content = content.replace(/border-gray-[12]00(?:\/[0-9]+)?/g, 'border-[var(--bg-4)]');
    content = content.replace(/border-gray-50(?:\/[0-9]+)?/g, 'border-[var(--bg-4)]');

    fs.writeFileSync(filePath, content);
}

function walkDir(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const fullPath = path.join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
            walkDir(fullPath);
        } else if (fullPath.endsWith('.tsx') || fullPath.endsWith('.ts')) {
            processFile(fullPath);
        }
    }
}

walkDir(widgetsDir);
console.log('Theme conversion completed for widgets.');
