import distutils
from distutils.core import setup
import py2exe


setup(options={"py2exe":{"dll_excludes":["MSVCP90.dll","MSVFW32.dll",
                 "AVIFIL32.dll",
                 "AVICAP32.dll",
                 "ADVAPI32.dll",
                 "CRYPT32.dll",
                 "WLDAP32.dll"]}},console=['Main.py'])

