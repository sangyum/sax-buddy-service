declare module "essentia.js" {
  export class EssentiaWASM {
    initialize(): Promise<void>;
  }

  export class Essentia {
    constructor(wasmModule: EssentiaWASM);
    version?: string;
    algorithmNames?: string;
    
    // Use Record<string, unknown> for method parameters and return types
    // This is more flexible and avoids complex interface issues
    [key: string]: unknown;
  }
}