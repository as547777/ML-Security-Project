
class GlobalHandler:
    def __init__(self, handlers=None):
        if handlers is not None:
            self.handlers=handlers
        else:
            self.handlers=[]
    
    def register(self, handler):
        self.handlers.append(handler)
    
    def handle(self, context):
        for handler in self.handlers:
            handler.handle(context)
        
        return context
        