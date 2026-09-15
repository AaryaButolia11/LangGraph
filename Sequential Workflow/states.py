# so we need to create a graph
# first thing we need to do is to create a state

import os

#first type -> TypedDict (Most commonly used) -> good in type checking at compile time
from typing import TypedDict

class State(TypedDict):
    topic: str
    summary: str
    score: int
     
# PyDantic model -> good in data validation and type checking at runtime
from pydantic import BaseModel, field_validator
class StateModel(BaseModel):
    topic: str
    summary: str
    score: int
    
    @field_validator('score')
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("Score must be a positive integer")
        return v
    
    
# Dataclass -> good in data validation and type checking at runtime
from dataclasses import dataclass, field
@dataclass
class State:
    topic: str =""
    summary: str =""
    score: int = field(default=0)
    
    def __post_init__(self):
        if self.score < 0:
            raise ValueError("Score must be a positive integer")
        
        
# by langGraph

from langgraph.graph import MessagesState
class State(MessagesState):
    topic: str
    summary: str
    score: int
    
    def __init__(self, topic:str, summary:str, score:int):
        self.topic = topic
        self.summary = summary
        self.score = score
        
        if self.score < 0:
            raise ValueError("Score must be a positive integer")