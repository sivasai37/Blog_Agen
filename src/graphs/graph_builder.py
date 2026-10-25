from langgraph.graph import StateGraph, START, END

from states.blogstate import BlogState
from src.nodes.blog__node import BlogNode


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm

    def build_topic_graph(self):
        graph = StateGraph(BlogState)
        blog_node = BlogNode(self.llm)

        graph.add_node(
            "title_creation",
            blog_node.title_creation
        )

        graph.add_node(
            "content_generation",
            blog_node.content_generation
        )

        graph.add_edge(START, "title_creation")
        graph.add_edge("title_creation", "content_generation")
        graph.add_edge("content_generation", END)

        return graph

    def build_lang_graph(self):
        graph = StateGraph(BlogState)
        blog_node = BlogNode(self.llm)

        graph.add_node(
            "title_creation",
            blog_node.title_creation
        )

        graph.add_node(
            "content_generation",
            blog_node.content_generation
        )

        graph.add_node(
            "hindi_translation",
            blog_node.hindi_translation
        )

        graph.add_node(
            "french_translation",
            blog_node.french_translation
        )

        graph.add_edge(START, "title_creation")
        graph.add_edge("title_creation", "content_generation")

        graph.add_conditional_edges(
            "content_generation",
            blog_node.route_decision,
            {
                "hindi": "hindi_translation",
                "french": "french_translation"
            }
        )

        graph.add_edge("hindi_translation", END)
        graph.add_edge("french_translation", END)

        return graph

    def setup_graph(self, usecase):
        if usecase == "topic":
            graph = self.build_topic_graph()

        elif usecase == "lang":
            graph = self.build_lang_graph()

        else:
            raise ValueError(f"Unknown use case: {usecase}")

        return graph.compile()