from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel
from typing import List, Dict, Any, Optional  # ensure all are imported

# -------------------------------------------------------------------------
# Neo4j Connection
# -------------------------------------------------------------------------
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="test1234"
)

# -------------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------------
class Action(BaseModel):
    name: str

class ECode(BaseModel):
    code: str
    description: Optional[str] = None

class ProblemScenario(BaseModel):
    description: str

class Symptom(BaseModel):
    name: str


# -------------------------------------------------------------------------
# Neo4j Interface Tool
# -------------------------------------------------------------------------
class Neo4jTool:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def get_all_actions(self) -> List[Action]:
        """Return all actions from the knowledge graph."""
        query = "MATCH (a:Action) RETURN a.name AS name"
        results = self.graph.query(query)
        return [Action(name=r["name"]) for r in results]

    def get_all_error_codes(self) -> List[ECode]:
        """Return all error codes from the knowledge graph."""
        query = "MATCH (e:ECode) RETURN e.code AS code, e.description AS description"
        results = self.graph.query(query)
        return [ECode(code=r["code"], description=r.get("description")) for r in results]

    def get_all_problem_scenarios(self) -> List[ProblemScenario]:
        """Return all problem scenarios."""
        query = "MATCH (p:ProblemScenario) RETURN p.description AS description"
        results = self.graph.query(query)
        return [ProblemScenario(description=r["description"]) for r in results]

    def get_all_symptoms(self) -> List[Symptom]:
        """Return all symptoms."""
        query = "MATCH (s:Symptom) RETURN s.name AS name"
        results = self.graph.query(query)
        return [Symptom(name=r["name"]) for r in results]

    def get_neighbors(self, label: str, identifier: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve all neighboring nodes connected to a given entity.
        label: one of ['Action', 'ECode', 'ProblemScenario', 'Symptom']
        identifier: property value (name, code, or description)
        """
        property_key = {
            "Action": "name",
            "ECode": "code",
            "ProblemScenario": "description",
            "Symptom": "name"
        }.get(label)

        if not property_key:
            raise ValueError(f"Unknown label: {label}")

        query = f"""
        MATCH (n:{label} {{{property_key}: $identifier}})-[r]-(m)
        RETURN type(r) AS rel_type, labels(m) AS node_labels, m AS node
        """
        results = self.graph.query(query, {"identifier": identifier})

        neighbors: Dict[str, List[Dict[str, Any]]] = {}
        for record in results:
            rel_type = record["rel_type"]
            neighbors.setdefault(rel_type, []).append({
                "labels": record["node_labels"],
                "properties": record["node"]
            })
        return neighbors


# -------------------------------------------------------------------------
# Helper Functions (wrapped as tools)
# -------------------------------------------------------------------------
neo_tool = Neo4jTool(graph)

def possible_actions_list() -> List[Action]:
    """List all possible actions from the graph."""
    return neo_tool.get_all_actions()

def possible_error_codes_list() -> List[ECode]:
    """List all error codes from the graph."""
    return neo_tool.get_all_error_codes()

def possible_scenarios_list() -> List[ProblemScenario]:
    """List all problem scenarios."""
    return neo_tool.get_all_problem_scenarios()

def possible_symptom_list() -> List[Symptom]:
    """List all symptoms."""
    return neo_tool.get_all_symptoms()

def get_entity_neighbors(label: str, identifier: str) -> Dict[str, Any]:
    """Get all neighbors of a specific entity."""
    return neo_tool.get_neighbors(label, identifier)

def propose_correct_action(correct_action: str) -> Action:
    """Propose the correct troubleshooting action."""
    return Action(name=correct_action)


# -------------------------------------------------------------------------
# Agent Configuration
# -------------------------------------------------------------------------
confirmation_criteria = "Do you want to proceed with this action?"

root_agent = Agent(
    model='gemini-2.5-flash',
    name='technician_agent',
    description="Troubleshoots broken dishwashers",
    instruction="""
    You are a helpful assistant that helps to troubleshoot a broken dishwasher: the Bosch 800 series.
    Ask the user troubleshoot questions until you are certain of the correct action.
    Correct action must be an action from the possible actions list.
    When you are certain, propose a correct action and wait for user confirmation.
    Literally state the proposed action. Example: 'proposed action: replace_motor'.
    """,
    tools=[
        FunctionTool(possible_actions_list),
        FunctionTool(possible_error_codes_list),
        FunctionTool(possible_scenarios_list),
        FunctionTool(possible_symptom_list),
        FunctionTool(get_entity_neighbors),
        FunctionTool(propose_correct_action, require_confirmation=confirmation_criteria)
    ],
)
