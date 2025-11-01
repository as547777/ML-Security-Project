
class GlobalHandler:
    def __init__(self, handlers=None):
        if handlers is not None:
            self.handlers=handlers
        else:
            handlers=[]
    
    def register(self, handler):
        self.handlers.append(handler)
    
    def handle(self, context):
        for handler in self.handlers:
            handler.handle(context)
        
        return context
        