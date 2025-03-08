/**
 * TypeScript client for the Toxicity Classification API
 */

// Types definition
interface ToxicityPrediction {
    prediction: boolean;
    probability: number;
  }
  
  interface ToxicityResult {
    TOXICITY: ToxicityPrediction;
    SEVERE_TOXICITY: ToxicityPrediction;
    INSULT: ToxicityPrediction;
    PROFANITY: ToxicityPrediction;
    IDENTITY_ATTACK: ToxicityPrediction;
    THREAT: ToxicityPrediction;
    NOT_TOXIC: ToxicityPrediction;
  }
  
  interface BatchToxicityResponse {
    results: {
      [key: string]: ToxicityResult;
    };
  }
  
  class ToxicityClassifier {
    private apiUrl: string;
  
    constructor(apiUrl: string = 'http://localhost:8000') {
      this.apiUrl = apiUrl;
    }
  
    /**
     * Classify a single text for toxicity
     * @param text The text to classify
     * @returns Promise with toxicity classification results
     */
    async classifyText(text: string): Promise<ToxicityResult> {
      try {
        const response = await fetch(`${this.apiUrl}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text }),
        });
  
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(`API error: ${errorData.detail || response.statusText}`);
        }
  
        return await response.json() as ToxicityResult;
      } catch (error) {
        console.error('Error classifying text:', error);
        throw error;
      }
    }
  
    /**
     * Classify multiple texts for toxicity
     * @param texts Array of texts to classify
     * @returns Promise with batch toxicity classification results
     */
    async classifyTexts(texts: string[]): Promise<BatchToxicityResponse> {
      try {
        const response = await fetch(`${this.apiUrl}/predict_batch`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ texts }),
        });
  
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(`API error: ${errorData.detail || response.statusText}`);
        }
  
        return await response.json() as BatchToxicityResponse;
      } catch (error) {
        console.error('Error classifying texts:', error);
        throw error;
      }
    }
  
    /**
     * Check if the API is healthy and ready to process requests
     * @returns Promise with health status
     */
    async checkHealth(): Promise<{ status: string; device?: string }> {
      try {
        const response = await fetch(`${this.apiUrl}/health`);
        
        if (!response.ok) {
          throw new Error(`Health check failed: ${response.statusText}`);
        }
  
        return await response.json();
      } catch (error) {
        console.error('Health check error:', error);
        throw error;
      }
    }
  }
  
  // Example usage
  async function example() {
    const classifier = new ToxicityClassifier('http://localhost:8000');
    
    try {
      // Check if API is ready
      const health = await classifier.checkHealth();
      console.log('API health status:', health);
      
      if (health.status === 'ready') {
        // Classify a single text
        const result = await classifier.classifyText('This is a test message');
        console.log('Single text classification:', result);
        
        // Classify multiple texts
        const batchResults = await classifier.classifyTexts([
          'This is the first message',
          'This is the second message'
        ]);
        console.log('Batch classification:', batchResults);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  }
  
  export { ToxicityClassifier, ToxicityResult, BatchToxicityResponse };