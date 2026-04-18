import rcssmin
import rjsmin
import os
import json
import time

# Configuration
VERSION_FILE = "version.json"
STATIC_DIR = "static"
DIST_DIR = os.path.join(STATIC_DIR, "dist")
FILES_TO_PROCESS = {
    "css": ["ckd_dashboard.css", "style.css"],
    "js": ["ckd_dashboard.js", "script.js"]
}

def load_version():
    with open(VERSION_FILE, "r") as f:
        return json.load(f)

def minify_assets():
    print("🚀 Starting Professional Asset Minification...")
    version_data = load_version()
    ver = version_data.get("app_version", "1.0.0")
    
    os.makedirs(DIST_DIR, exist_ok=True)
    
    # Header for minified files
    header = f"/* MediScan AI v{ver} | (c) 2026 MediScan Research Lab | Built: {time.ctime()} */\n"

    # Process CSS
    for css_file in FILES_TO_PROCESS["css"]:
        src_path = os.path.join(STATIC_DIR, css_file)
        dist_name = css_file.replace(".css", ".min.css")
        dist_path = os.path.join(DIST_DIR, dist_name)
        
        if os.path.exists(src_path):
            with open(src_path, "r") as f:
                content = f.read()
            minified = rcssmin.cssmin(content)
            with open(dist_path, "w") as f:
                f.write(header + minified)
            print(f"  ✅ Minified CSS: {css_file} -> dist/{dist_name}")

    # Process JS
    for js_file in FILES_TO_PROCESS["js"]:
        src_path = os.path.join(STATIC_DIR, js_file)
        dist_name = js_file.replace(".js", ".min.js")
        dist_path = os.path.join(DIST_DIR, dist_name)
        
        if os.path.exists(src_path):
            with open(src_path, "r") as f:
                content = f.read()
            minified = rjsmin.jsmin(content)
            with open(dist_path, "w") as f:
                f.write(header + minified)
            print(f"  ✅ Minified JS: {js_file} -> dist/{dist_name}")

    print("✨ Build Complete.")

if __name__ == "__main__":
    minify_assets()
