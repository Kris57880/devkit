#!/usr/bin/env python3
import sys
from zipfile import ZipFile

FILES_ZIP = sys.argv[1]
with ZipFile(FILES_ZIP) as zipfile:
	zipfile.extractall()
