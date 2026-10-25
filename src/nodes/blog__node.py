from langchain_core.prompts import ChatPromptTemplate

from states.blogstate import BlogState, Blog


class BlogNode:
    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        topic = state.get("topic")

        if not topic:
            raise ValueError("Topic is required")

        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert blog writer.
            Create an engaging title for a blog about: {topic}
            Return only the title.
            """
        )

        response = (prompt | self.llm).invoke({"topic": topic})

        return {
            "blog": Blog(
                title=response.content.strip(),
                content=""
            )
        }

    def content_generation(self, state: BlogState):
        blog = state.get("blog")

        if not blog:
            raise ValueError("Blog title is required")

        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert blog writer.
            Write an informative blog based on this title:

            {title}

            Return only the blog content.
            """
        )

        response = (prompt | self.llm).invoke(
            {"title": blog.title}
        )

        return {
            "blog": Blog(
                title=blog.title,
                content=response.content.strip()
            )
        }

    def route_decision(self, state: BlogState):
        language = state.get("current_language", "").strip().lower()

        if language == "hindi":
            return "hindi"

        if language == "french":
            return "french"

        raise ValueError(
            "current_language must be Hindi or French"
        )

    def hindi_translation(self, state: BlogState):
        return self._translate_blog(state, "Hindi")

    def french_translation(self, state: BlogState):
        return self._translate_blog(state, "French")

    def _translate_blog(self, state: BlogState, language: str):
        blog = state.get("blog")

        if not blog:
            raise ValueError("Blog content is required for translation")

        translator = self.llm.with_structured_output(Blog)

        translated_blog = translator.invoke(
            f"""
            Translate the following blog into {language}.
            Translate both the title and content.
            Preserve the meaning and details.

            Title:
            {blog.title}

            Content:
            {blog.content}
            """
        )

        return {"blog": translated_blog}