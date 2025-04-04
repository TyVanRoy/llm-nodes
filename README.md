# llm-nodes

A lightweight, composable TypeScript library for working with LLMs that extends LangChain with a simpler, more intuitive API.

## Installation

```bash
npm install llm-nodes
```

## Environment Variables

This library uses dotenv to load API keys from your environment. Create a `.env` file in the root of your project with your API keys:

```
# API Keys for different providers
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROK_API_KEY=your_grok_api_key_here
# Add others as needed
```

## Features

-   **Simplified Node Pattern**: Combines prompt templates, LLM configuration, and response parsing into a cohesive unit
-   **Type-Safe**: Full TypeScript support with generics for input and output types
-   **Composable**: Easily connect nodes using functional composition
-   **Provider Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, etc.)
-   **Specialized Nodes**: Purpose-built nodes for common tasks like classification, extraction, and RAG
-   **Flexible Pipelines**: Advanced pipeline patterns for complex workflows
-   **LangChain Compatible**: Built on top of LangChain for compatibility with its ecosystem
-   **Lightweight**: Minimal API with sensible defaults for rapid development

## Quick Start

```typescript
import { LLMNode, jsonParser } from "llm-nodes";

// Create a simple node for sentiment analysis
const sentimentAnalyzer = new LLMNode<
    { text: string },
    { sentiment: string; score: number }
>({
    promptTemplate: `
    Analyze the sentiment of the following text:
    {{text}}
    
    Respond with a JSON object containing:
    - sentiment: either "positive", "negative", or "neutral"
    - score: a number from -1 (very negative) to 1 (very positive)
  `,
    llmConfig: {
        provider: "openai",
        model: "gpt-3.5-turbo",
        temperature: 0.3,
    },
    parser: jsonParser<{ sentiment: string; score: number }>(),
});

// Use the node
async function analyzeSentiment(text: string) {
    const result = await sentimentAnalyzer.execute({ text });
    console.log(result); // { sentiment: "positive", score: 0.8 }
    return result;
}

analyzeSentiment("I'm having a fantastic day today!");
```

## Node Types

### Core LLM Node

The foundation of the library, encapsulating prompt templates, LLM configuration, and response parsing:

```typescript
const summarizer = new LLMNode<{ text: string }, string>({
    promptTemplate: "Summarize the following text: {{text}}",
    llmConfig: {
        provider: "anthropic",
        model: "claude-3-opus-20240229",
    },
    parser: (text) => text,
});
```

### TextNode

A simplified node for text generation with the text parser built-in:

```typescript
const textGenerator = new TextNode({
    promptTemplate: "Write a short story about {{topic}} in {{style}} style.",
    llmConfig: {
        provider: "openai",
        model: "gpt-4",
    },
});

// Use it
const story = await textGenerator.execute({
    topic: "a robot learning to paint",
    style: "magical realism",
});

// Add instructions without creating a new node
const detailedGenerator = textGenerator.withAdditionalPrompt(
    "Include vivid sensory details and a surprising twist at the end."
);
```

### Specialized LLM Nodes

The library includes several specialized node types for common tasks:

-   **StructuredOutputNode**: Enforces output schema with Zod validation
-   **ClassificationNode**: Classifies inputs into predefined categories
-   **ExtractionNode**: Extracts structured fields from text
-   **ChainNode**: Implements multi-step reasoning chains
-   **RAGNode** (Incomplete): Retrieval-augmented generation with document context

```typescript
// Example: Classification node
const categoryClassifier = new ClassificationNode({
    categories: ["business", "technology", "health", "entertainment"] as const,
    llmConfig: { provider: "openai", model: "gpt-3.5-turbo" },
    includeExplanation: true,
});

// Example: Extraction node
const contactExtractor = new ExtractionNode({
    fields: [
        { name: "name", description: "Full name of the person" },
        { name: "email", description: "Email address", format: "email" },
        { name: "phone", description: "Phone number", required: false },
    ],
    promptTemplate: "Extract contact information from: {{text}}",
    llmConfig: { provider: "anthropic", model: "claude-3-sonnet-20240229" },
});
```

### Utility Nodes

Non-LLM nodes for pipeline manipulation:

-   **DataEnricherNode**: Injects external data into the pipeline
-   **MergeNode**: Combines outputs from multiple nodes

```typescript
// Inject site map data
const siteMapEnricher = new DataEnricherNode({
    enricher: (article, siteMap) => ({ article, siteMap }),
    context: {
        pages: [
            /* site pages */
        ],
    },
});

// Merge parallel processing results
const merger = new MergeNode({
    merger: ([summary, keyPoints]) => ({ summary, keyPoints }),
});
```

## Pipeline Patterns

### Simple Linear Pipeline

Chain nodes together with the `pipe()` method:

```typescript
const pipeline = extractor.pipe(enricher).pipe(generator);
const result = await pipeline.execute(input);
```

### Data Enrichment Pipeline

Inject external data into your pipeline:

```typescript
// Create a pipeline with external data
const keyPointExtractor = new ExtractionNode({
    /*...*/
});

const siteMapEnricher = new DataEnricherNode({
    enricher: (extractionResult, siteMap) => ({
        extraction: extractionResult,
        siteMap: siteMap,
    }),
    context: fetchSiteMap, // async function or static data
});

const articleFormatter = new LLMNode({
    /*...*/
});

const pipeline = keyPointExtractor.pipe(siteMapEnricher).pipe(articleFormatter);
```

### Parallel Processing Pipeline

Process data through multiple nodes and merge the results:

```typescript
// Define parallel nodes
const summaryNode = new LLMNode({
    /*...*/
});
const keyPointsNode = new LLMNode({
    /*...*/
});

// Merge node to combine results
const mergeNode = new MergeNode({
    merger: ([summaryResult, keyPointsResult]) => ({
        summary: summaryResult.summary,
        keyPoints: keyPointsResult.keyPoints,
    }),
});

// Helper to create a parallel pipeline
const parallelPipeline = MergeNode.createPipeline(
    [summaryNode, keyPointsNode],
    mergeNode
);

// Use the pipeline
const mergedResult = await parallelPipeline({ text: "..." });
```

### Custom Execution Flow

For maximum flexibility, use `execute()` directly:

```typescript
async function customWorkflow(text) {
    // First analysis
    const analysis = await analyzer.execute({ text });

    // Custom business logic
    if (analysis.sentiment === "negative") {
        // Handle negative content specially
    }

    // External data integration
    const userData = await fetchUserData();

    // Final generation with combined context
    return generator.execute({
        topic: analysis.mainTopic,
        userData,
    });
}
```

## API Reference

### `LLMNode<TInput, TOutput>`

The core class that encapsulates an LLM interaction pattern.

#### Constructor Options

```typescript
{
  promptTemplate: string | ((input: TInput) => string);
  llmConfig: {
    provider: string;  // 'openai', 'anthropic', etc.
    model: string;
    temperature?: number;
    maxTokens?: number;
    systemPrompt?: string;
    // Provider-specific options
  };
  parser: (rawResponse: string) => TOutput;
}
```

#### Methods

-   `execute(input: TInput): Promise<TOutput>` - Execute the node with input data
-   `pipe<TNextOutput>(nextNode: IExecutable<TOutput, TNextOutput>): IExecutable<TInput, TNextOutput>` - Connect to another node

### Specialized Nodes

-   `StructuredOutputNode<TInput, TOutput>`: Schema-validated outputs using Zod
-   `ClassificationNode<TInput, TCategory>`: Classification with predefined categories
-   `ExtractionNode<TInput, TOutput>`: Field extraction from unstructured text
-   `ChainNode<TInput, TOutput>`: Multi-step reasoning chains
-   `RAGNode<TInput, TOutput>` (Incomplete): Retrieval-augmented generation

### Utility Nodes

-   `DataEnricherNode<TInput, TOutput>`: Inject external data into pipelines
-   `MergeNode<TInputs, TOutput>`: Combine outputs from multiple nodes

### Parsers

-   `jsonParser<T>()` - Parse JSON responses
-   `jsonFieldParser<T>(field: string)` - Extract a specific field from JSON
-   `regexParser<T>(patterns: Record<keyof T, RegExp>)` - Extract data using regex patterns
-   `labeledFieldsParser<T>()` - Parse responses with labeled fields
-   `textParser()` - Return raw text responses

## Token Usage Tracking

The library provides built-in token usage tracking:

```typescript
// Create an LLM node
const textGenerator = new TextNode({
    promptTemplate: "Write about {{topic}}",
    llmConfig: {
        provider: "anthropic",
        model: "claude-3-sonnet-20240229",
    },
});

// Use the node
const result = await textGenerator.execute({ topic: "AI" });

// Get token usage statistics
const usage = textGenerator.getTotalTokenUsage();
console.log(`Input Tokens: ${usage.inputTokens}`);
console.log(`Output Tokens: ${usage.outputTokens}`);
console.log(`Total Tokens: ${usage.totalTokens}`);

// Get detailed usage records
const records = textGenerator.getUsageRecords();
```

Token tracking also works across pipelines:

```typescript
// Create a pipeline
const pipeline = nodeA.pipe(nodeB).pipe(nodeC);

// Execute
const result = await pipeline.execute(input);

// Get token usage for the entire pipeline
const usage = pipeline.getTotalTokenUsage();
```

## TODOs

- Support for extended thinking modes
- RAGNode implementation

## License

MIT
