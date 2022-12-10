# Threedle Logger: Thlogger

import polyscope as ps
import os

LOG_NONE = 0
LOG_INFO = 1
LOG_DEBUG = 3
LOG_TRACE = 5

VIZ_NONE = 7
VIZ_INFO = 10
VIZ_DEBUG = 13
VIZ_TRACE = 15

thlog_prefixes = {
      LOG_INFO: "INFO "
    , LOG_DEBUG: "DEBUG"
    , LOG_TRACE: "TRACE"
}

class Thlogger:
    def __init__(self, loglevel, vizlevel, moduleprefix, imports=[], propagate_to_imports=True):
        self.loglevel = loglevel
        self.vizlevel = vizlevel if os.environ.get("NO_POLYSCOPE") is None else VIZ_NONE
        self.moduleprefix = moduleprefix
        self.imported_thloggers = imports
        self.ps_initialized = False
        if propagate_to_imports:
            for thlogger in self.imported_thloggers:
                thlogger.loglevel = self.loglevel
                thlogger.vizlevel = self.vizlevel
                
    def set_levels(self, loglevel, vizlevel, propagate_to_imports=False):
        self.loglevel = loglevel
        self.vizlevel = vizlevel
        if propagate_to_imports:
            for thlogger in self.imported_thloggers:
                thlogger.loglevel = self.loglevel
                thlogger.vizlevel = self.vizlevel

    def init_polyscope(self, propagate_to_imports=True):
        if self.vizlevel == VIZ_NONE:
            # when NO_POLYSCOPE=1 env var is used, that often also means 
            # polyscope is not available so don't init at all
            return
        if not self.ps_initialized:
            ps.init()
            self.ps_initialized = True
            # broadcast to other thloggers that ps has been initialized
            for thlogger in self.imported_thloggers:
                thlogger.ps_initialized = True

    def log(self, loglevel, message):
        if self.loglevel >= loglevel:
            if loglevel not in thlog_prefixes:
                prefix = "  !  "
            else:
                prefix = thlog_prefixes[loglevel]
            print(f"[{self.moduleprefix} | {prefix}] {message}")
    
    def info(self, msg):
        return self.log(LOG_INFO, msg)
    def debug(self, msg):
        return self.log(LOG_DEBUG, msg)
    def trace(self, msg):
        return self.log(LOG_TRACE, msg)
    
    def __call__(self, *args):
        return self.log(*args)
    
    def do(self, vizlevel, fn, needs_polyscope = False):
        if self.vizlevel >= vizlevel:
            if (not needs_polyscope) or \
                (needs_polyscope and self.ps_initialized):
                return fn()

    def err(self, message):
        print(f"[{self.moduleprefix} | ERROR] {message}")
    
    def guard(self, vizlevel, needs_polyscope = False):
        return (self.vizlevel >= vizlevel) and \
            ((not needs_polyscope) or (needs_polyscope and self.ps_initialized))