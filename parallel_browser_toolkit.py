import os
import asyncio
from langchain_openai import ChatOpenAI
from agno.tools import Toolkit
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

class ParallelBrowserToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="parallel_browser_toolkit")
        self.llm = ChatOpenAI(model='gpt-4o')
        self.register(self.run_parallel_tasks)

    async def _run_single_task(self, task_description: str):
        try:
            context_config = BrowserContextConfig(
                wait_for_network_idle_page_load_time=10000,
                highlight_elements=False,
                viewport_expansion=500,
                _force_keep_context_alive=True,
                save_downloads_path=os.path.join(os.path.expanduser('~'), 'downloads')
            )
            browser_config = BrowserConfig(
                headless=False,
                disable_security=True,
                cdp_url="http://localhost:9222",
                new_context_config=context_config,
                _force_keep_browser_alive=True,
            )
            browser = Browser(config=browser_config)
            browser_context = await browser.new_context()
            await browser_context.create_new_tab("about:blank")
            agent = Agent(
                llm=self.llm,
                task=task_description,
                browser_context=browser_context
            )
            result = await agent.run()
            await browser_context.close()
            return str(result)
        except Exception as e:
            return f"Task failed: {str(e)}"

    async def run_parallel_tasks(self, task_descriptions: list):
        """
        Run multiple browser tasks in parallel. Each task gets its own context.
        Args:
            task_descriptions (list): List of task descriptions (e.g., URLs or prompts)
        Returns:
            List of results (one per task)
        """
        tasks = [self._run_single_task(desc) for desc in task_descriptions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results 