class PerpetualCoordinator:
    """ORIGINAL coordinator + x402 treasury"""
    
    def __init__(self, deepseek_client, treasury=None):  # ADD treasury
        self.deepseek = deepseek_client
        self.treasury = treasury  # ADD x402
        self.queue_manager = ResearchQueueManager()
        self.agents = []
    
    def initialize_agents(self):
        """ORIGINAL agent setup + x402 treasury"""
        queues = self.queue_manager.queues
        
        self.agents = [
            # Same 7 agents, just add treasury parameter
            QueryGeneratorAgent(self.deepseek, queues['query'], self.treasury),
            ResearchAgent(self.deepseek, queues['query'], queues['research'], self.treasury),
            InterpreterAgent(self.deepseek, queues['research'], queues['interpretation'], self.treasury),
            QualityAgent(self.deepseek, queues['interpretation'], queues['quality'], self.treasury),
            ResearchAmplifierAgent(self.deepseek, queues['quality'], queues['amplification'], self.treasury),
            BuilderAgent(self.deepseek, queues['amplification'], queues['building'], self.treasury),
            EvaluatorAgent(self.deepseek, queues['building'], queues['evaluation'], self.treasury)
        ]