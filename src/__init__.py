#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OfficeCowork - AI-powered intelligent code generation and autonomous task execution system

Copyright (c) 2025 OfficeCowork Research Group.
Licensed under the Apache License, Version 2.0

For explicit imports, use:
    from src.main import OfficeCoworkClient, create_client, OfficeCoworkMain
    from src.config_loader import load_config
    from src.tool_executor import ToolExecutor
"""

# Removed module-level imports to avoid loading heavy dependencies at startup
# Users should import explicitly from submodules as needed

__version__ = "1.2.0"
__author__ = "OfficeCowork Team"
__email__ = "contact@agia.ai"
__description__ = "AI Code Auto-Generator - Intelligent code generation and task execution system based on Claude Sonnet"

# Public API
__all__ = [
    'OfficeCoworkClient',
    'create_client', 
    'OfficeCoworkMain',
    '__version__',
    '__author__',
    '__email__',
    '__description__'
] 