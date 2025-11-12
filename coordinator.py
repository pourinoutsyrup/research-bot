import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Use your actual DeepSeek class
from workflows.ai_strategy_extractor import DeepSeekExtractor
from swarm.agents import AGENT_REGISTRY, BaseResearchAgent, ResearchTask

logger = logging.getLogger(__name__)

@dataclass
class ResearchResult:
    concept: str
    mathematical_formulation: str
    strategy_code: str
    backtest_results: Dict[str, Any]
    confidence_score: float
    source: str
    discovered_at: datetime

class AthenaCoordinator:
    def __init__(self, deepseek_client, config: Dict[str, Any]):
        self.deepseek = deepseek_client
        self.config = config
        self.agents = {}
        self.results = []
        self.concept_memory = []
        
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all agents using your actual agent classes"""
        try:
            # Initialize all your agents
            self.agents['universal_scout'] = AGENT_REGISTRY['universal_scout'](
                'universal_scout', self.deepseek
            )
            self.agents['interpreter'] = AGENT_REGISTRY['interpreter'](
                'interpreter', self.deepseek
            )
            self.agents['builder'] = AGENT_REGISTRY['builder'](
                'builder', self.deepseek
            )
            self.agents['evaluator'] = AGENT_REGISTRY['evaluator'](
                'evaluator', self.deepseek
            )
            self.agents['quality'] = AGENT_REGISTRY['quality'](
                'quality', self.deepseek
            )
            self.agents['analyst'] = AGENT_REGISTRY['analyst'](
                'analyst', self.deepseek
            )
            
            logger.info(f"✅ Initialized {len(self.agents)} Athena agents: {list(self.agents.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise

    async def run_research_cycle(self, queries: List[str]) -> List[ResearchResult]:
        """Execute full Athena research pipeline using your agents"""
        results = []
        total_queries = len(queries)
        
        logger.info(f"🔬 Starting research cycle with {total_queries} queries")
        
        for i, query in enumerate(queries):
            try:
                logger.info(f"🎯 [{i+1}/{total_queries}] Processing: {query}")
                
                # Phase 1: Discovery with universal scout
                logger.info(f"   Phase 1: Discovery...")
                discovered = await self._discovery_phase(query)
                
                # Phase 2: Interpretation
                logger.info(f"   Phase 2: Interpretation...")
                interpreted = await self._interpretation_phase(discovered)
                
                # Phase 3: Quality check
                logger.info(f"   Phase 3: Quality assessment...")
                quality_checked = await self._quality_phase(interpreted)
                
                # FIX: Ensure we extend with valid results
                if quality_checked:
                    results.extend(quality_checked)
                    logger.info(f"   ✅ Completed: {len(quality_checked)} validated concepts")
                else:
                    logger.info(f"   ⚠️ Completed: 0 validated concepts (quality threshold not met)")
                
            except Exception as e:
                logger.error(f"❌ Research cycle failed for {query}: {e}")
                continue
        
        # FIX: Always return a list, even if empty
        return results or []

    async def _discovery_phase(self, query: str) -> List[Dict]:
        """Universal scout discovers concepts"""
        try:
            task = ResearchTask('universal_scout', query, priority=1)
            result = await self.agents['universal_scout'].execute(task)
            return [result] if result.get('success') else []
        except Exception as e:
            logger.error(f"Discovery phase failed: {e}")
            return []

    async def _interpretation_phase(self, concepts: List[Dict]) -> List[Dict]:
        """Interpreter makes concepts actionable - OPTIMIZED VERSION"""
        interpreted = []
        for concept in concepts:
            try:
                # OPTIMIZATION: Extract only technique names, not full JSON
                discovered_data = concept.get('discovered_concepts', {})
                strategy_code = discovered_data.get('strategy_code', '{}')
                
                # Parse and extract just the technique names for interpretation
                try:
                    techniques_data = json.loads(strategy_code)
                    if 'volatility_modeling_techniques' in techniques_data:
                        # Take only the first 2 techniques to keep payload small
                        techniques = techniques_data['volatility_modeling_techniques'][:2]
                        technique_names = [tech.get('technique_name', '') for tech in techniques]
                        interpretation_input = "Techniques: " + ", ".join(technique_names)
                    else:
                        interpretation_input = concept.get('task', 'Unknown concept')[:200]  # Limit length
                except:
                    interpretation_input = concept.get('task', 'Unknown concept')[:200]  # Limit length
                
                logger.info(f"   Sending to interpreter: {interpretation_input[:100]}...")
                task = ResearchTask('interpreter', interpretation_input)
                result = await self.agents['interpreter'].execute(task)
                
                if result.get('success'):
                    interpreted.append({
                        'original_concept': concept,
                        'interpretation': result
                    })
                    
            except Exception as e:
                logger.error(f"Interpretation failed: {e}")
                continue
        return interpreted

    async def _quality_phase(self, concepts: List[Dict]) -> List[ResearchResult]:
        """Quality assessment for human prioritization - NO FILTERING"""
        validated = []
        for concept in concepts:
            try:
                original_task = concept.get('original_concept', {}).get('task', 'Unknown')
                
                # Get quality assessment (for scoring only)
                assessment_input = f"Score tradability of: {original_task}"
                
                logger.info(f"   Quality scoring: {original_task}")
                task = ResearchTask('quality', assessment_input)
                quality_result = await self.agents['quality'].execute(task)
                
                interpretation_data = concept.get('interpretation', {}).get('interpretation', {})
                
                if quality_result and quality_result.get('success'):
                    assessment = quality_result.get('quality_assessment', {})
                    
                    # Extract scores for confidence calculation
                    tradability_score = assessment.get('tradability_score', 5.0)
                    ease_score = assessment.get('ease_score', 5.0)
                    novelty_score = assessment.get('novelty_score', 5.0)
                    reasoning = assessment.get('reasoning', 'No reasoning')
                    
                    # Calculate overall confidence (weighted average)
                    confidence = min((tradability_score * 0.5 + ease_score * 0.3 + novelty_score * 0.2) / 10.0, 1.0)
                    
                    result = ResearchResult(
                        concept=original_task,
                        mathematical_formulation=str(interpretation_data),
                        strategy_code="",
                        backtest_results={},
                        confidence_score=confidence,
                        source="athena_pipeline",
                        discovered_at=datetime.now(),
                        # Store the individual scores for your review
                        quality_scores={
                            'tradability': tradability_score,
                            'ease': ease_score,
                            'novelty': novelty_score,
                            'reasoning': reasoning
                        }
                    )
                    validated.append(result)
                    logger.info(f"   ✅ SCORED: tradability={tradability_score}/10, confidence={confidence:.2f}")
                else:
                    # If assessment fails, proceed with medium confidence
                    result = ResearchResult(
                        concept=original_task,
                        mathematical_formulation=str(interpretation_data),
                        strategy_code="",
                        backtest_results={},
                        confidence_score=0.5,
                        source="athena_pipeline",
                        discovered_at=datetime.now(),
                        quality_scores={'error': 'Assessment failed'}
                    )
                    validated.append(result)
                    logger.info(f"   ⚠️ ASSESSMENT FAILED: proceeding with medium confidence")
                        
            except Exception as e:
                logger.error(f"Quality phase failed: {e}")
                # On error, proceed with low confidence
                result = ResearchResult(
                    concept=concept.get('original_concept', {}).get('task', 'Unknown'),
                    mathematical_formulation="Error in assessment",
                    strategy_code="",
                    backtest_results={},
                    confidence_score=0.3,
                    source="athena_pipeline",
                    discovered_at=datetime.now(),
                    quality_scores={'error': str(e)}
                )
                validated.append(result)
                    
        return validated
    
    async def run_research_cycle(self, queries: List[str]) -> List[ResearchResult]:
        """Execute full Athena research pipeline using your agents"""
        results = []
        total_queries = len(queries)
        
        logger.info(f"🔬 Starting research cycle with {total_queries} queries")
        
        for i, query in enumerate(queries):
            try:
                logger.info(f"🎯 [{i+1}/{total_queries}] Processing: {query}")
                
                # Phase 1: Discovery
                logger.info(f"   Phase 1: Discovery...")
                discovered = await self._discovery_phase(query)
                
                # Phase 2: Interpretation  
                logger.info(f"   Phase 2: Interpretation...")
                interpreted = await self._interpretation_phase(discovered)
                
                # Phase 3: Quality check (NOW AUTO-APPROVES)
                logger.info(f"   Phase 3: Quality assessment...")
                quality_checked = await self._quality_phase(interpreted)
                
                # Phase 4: Strategy Building
                logger.info(f"   Phase 4: Strategy building...")
                built_strategies = await self._building_phase(quality_checked)
                
                if built_strategies:
                    results.extend(built_strategies)
                    logger.info(f"   ✅ Completed: {len(built_strategies)} built strategies")
                else:
                    logger.info(f"   ⚠️ Completed: 0 built strategies")
                
            except Exception as e:
                logger.error(f"❌ Research cycle failed for {query}: {e}")
                continue
        
        # 🚨 CRITICAL: Return the results list
        return results  # THIS LINE MUST BE HERE