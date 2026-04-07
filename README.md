# Bible Retrieval

Intended use: To serve as an experimental platform for testing out how injecting biblical principles can help with morality. Specifically on the following datasets:

- Hendrycks ETHICS benchmark

# Sections:
1) Biblical embedding dataset
   This is storing every verse as a unique vector, every chapter as a unique vector, and every book as a unique vector, with the possibility of also including "themes" or "stories" as unique vectors, based on a determination of a sliding window threshold as well. 
2) Retrieval mechanism
   Pure retrieval based on RAG style methods is probably not ideal; reranking may be required. Using ColBERTv2 and advances such as MixedBread to help retrieve is helpful for proper knowledge of retrieval. 
3) Pure comparison
   This is comparing the raw model, with a generic system prompt, to the ethical question of interest. Something that can be done locally. But then, based on the comparison, also perform with proper injection based on the retrieval mechanism that was created in step 2). 




Inspired by :
`https://github.com/christian-machine-intelligence/psalm-alignment` - ??
`https://github.com/Lumi-node/hermeneutica` - Andrew Young



# Models to Use:
The models to use, and why, and when they were researched. 

Default hardware:
```
CPU: 12th Gen Intel® Core™ i7-12650H × 16
GPU: Nvidia GeForce RTX 3060 Laptop 6Gb
```

## Embedding model:
Model leaderboard: `https://huggingface.co/spaces/mteb/leaderboard`

### Supporting docs:
- Not all metrics are equal: https://modal.com/blog/mteb-leaderboard-article    