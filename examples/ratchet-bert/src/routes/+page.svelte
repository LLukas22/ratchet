<script>
import { AutoModel, AutoTokenizer } from '@xenova/transformers';
import { Model, DecodingOptionsBuilder, default as init, Quantization, Segment } from "@ratchet-ml/ratchet-web";

  /**
   * @type {Model | null}
   */
let model = null;

async function load_model(){

    await init();
    model = await Model.load( { Bert:"mini_l_m"} ,Quantization.F32, (/** @type {any} */ p) => console.log(p));
    console.log(model);

    // if(!tokenizer){
    //     console.log('loading tokenizer');
    //     tokenizer = await AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');
    // }

    // if(!model){
    //     console.log('loading model');
    //     model = await AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');
    // }
}

async function embed(){
    const text1 = 'Why did the crab cross the road?';
    const text2  = 'Another unrelated prompt';
    if (!model){
        console.log('no model loaded');
        return;
    }
    const tokens  = await model.run({texts: Array(32).fill(text1)});
    console.log([text1, text2]);
    console.log(tokens);
}

</script>

<h1>Welcome to SvelteKit</h1>

<p>
    <button on:click={load_model}>Load Model</button>
</p>

<p>
    <button on:click={embed}>Embed</button>
</p>
