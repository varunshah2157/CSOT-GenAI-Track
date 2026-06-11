**features:**
i have built a perplexity clone with the following features:
1) web search : searches the web using serper
2) web fetch : to get info from specific url
3) save research note : makes file in speicifc directory notes/
4) alpha xiv oauth authentification
5) alpha xiv connects my chatbot to academic repo of alpha xiv using mcp so it can look up research papers.
6) tui with shortcuts(as mentioned in build3)
7) split panel tui
8) streaming
9) good error handling



**mechanics of my agent:**

decides which model to use based on priority list in case some model isnt available.
i give some prompt to the chatbot and that gets added to message history.
model has access to message history and tools. It makes a decision whether it needs external data and decides to use corresponding tool.
tools are executed if needed.
since everything is in a for loop(with max 10 iterations for safety) there is a feedback loop in case model needs to use multiple tools.
output is given(there is streaming). tools used are displayed on right panel of tui.



**self feedback:**

i had a lot of fun in this weeks task. got to learn quite a lot of stuff which i didnt know before. i knew the basics of tools earlier but now my concepts r clearer. got to learn about web tools, alpha xiv, mcp... these took some time to read and understand. tui was one of my favorite parts lol because of how much cleaner and better it makes everything look. i like the final tui design and streaming feature of my chatbot even tho the tui is what was given by u guys i had to make a few changes to code and libraries used to preserve that clean ui when implementing streaming(took quite a lot of attempts). didnt really look at this in detail and just used code provided by senior but kinda got to know about oauth authentication when faced with the authentication issue of alpha xiv. i think i followed the given instruction pretty well and the bonus tasks too... but i did not add anything of my own. only thing i did kinda add was model priority list fallback function because when a model is unavailable it just crashed. also ofc i vibe coded most of this but did understand most of it too.