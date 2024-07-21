<script>
    // import { AutoModel, AutoTokenizer } from '@xenova/transformers';
    import { Model, DecodingOptionsBuilder, default as init, Quantization, Segment } from "@ratchet-ml/ratchet-web";
    import { loadHnswlib, syncFileSystem } from 'hnswlib-wasm';

    import { createRepo, uploadFiles, uploadFilesWithProgress, deleteFile, deleteRepo, listFiles, whoAmI } from "@huggingface/hub";

    /**
   * @type {import("hnswlib-wasm").HnswlibModule | null}
   */
    let hnswlib = null;

    /**
     * @type {Model | null}
     */
    let model = null;

    /**
   * @param {string} name
   * @param {number} dimension
   */
    async function create_index(name,dimension){
        if (!hnswlib){
            console.debug('Loading hnswlib...');
            hnswlib = await loadHnswlib();
            hnswlib.EmscriptenFileSystemManager.setDebugLogs(true);
        }

        let index = new hnswlib.HierarchicalNSW('cosine', dimension,name);
        await syncFileSystem('read');
        const exists = hnswlib.EmscriptenFileSystemManager.checkFileExists(name);
        if (!exists) {
            index.initIndex(100000, 48, 128, 100);
            index.setEfSearch(32);
            await index.writeIndex(name);
        } else {
            await index.readIndex(name, 100000);
            index.setEfSearch(32);
        }
    }


    async function load_model(){

        await init();
        model = await Model.load( { Bert:"mini_l_m"} ,Quantization.F32, (/** @type {any} */ p) => console.log(p));
        console.log(model);

        await create_index("foo.dat", 384);
      

        console.log(hnswlib);
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

        if (!hnswlib){
            console.log('no db loaded');
            return;
        }
}

</script>

<h1>Welcome to SvelteKit</h1>

<p>
    <button on:click={load_model}>Load Model</button>
</p>

<p>
    <button on:click={embed}>Embed</button>
</p>
