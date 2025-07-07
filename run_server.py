#!/usr/bin/env python3
"""
Development server startup script for Sax Buddy Service API
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        reload=True,
        reload_dirs=["src"],
        log_level="debug"
    )