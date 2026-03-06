# Sentiment Slang Analyzer - Project Flowchart

## Complete System Flow

```mermaid
flowchart TD
    Start([User Opens Browser]) --> LoadPage[Load Web Interface<br/>localhost:8000]
    
    LoadPage --> CheckModels{Models API<br/>/models}
    CheckModels --> LoadModelsJS[Load Available Models<br/>Version1 & Version2]
    
    LoadModelsJS --> ShowDropdown[Show Model Selector<br/>Default: No Selection]
    ShowDropdown --> DisabledState[Text Input: DISABLED<br/>Analyze Button: DISABLED<br/>Message: Select model first]
    
    DisabledState --> UserSelectModel{User Selects<br/>Model Version?}
    
    UserSelectModel -->|No Selection| DisabledState
    UserSelectModel -->|Version 1 Selected| EnableV1[Enable Text Input<br/>Show: Baseline Model Info]
    UserSelectModel -->|Version 2 Selected| EnableV2[Enable Text Input<br/>Show: Enhanced Model Info]
    
    EnableV1 --> EnterText[User Enters Text]
    EnableV2 --> EnterText
    
    EnterText --> ClickAnalyze{Click<br/>Analyze Button}
    
    ClickAnalyze --> SendAPI[Send POST Request<br/>/predict]
    SendAPI --> APIReceive[FastAPI Receives<br/>text + model_version]
    
    APIReceive --> CleanText[Clean Text<br/>preprocessing.py]
    CleanText --> LoadModel{Load Selected<br/>Model}
    
    LoadModel -->|Version 1| UseV1[Use Version 1<br/>Tokenizer + Model]
    LoadModel -->|Version 2| UseV2[Use Version 2<br/>Tokenizer + Model]
    
    UseV1 --> Tokenize[Tokenize Text]
    UseV2 --> Tokenize
    
    Tokenize --> PadSequence[Pad Sequences<br/>Max Length: 30]
    PadSequence --> Predict[Model Prediction<br/>Bi-LSTM Processing]
    
    Predict --> GetResult[Get Sentiment<br/>+ Confidence]
    GetResult --> ReturnJSON[Return JSON Response<br/>sentiment, confidence,<br/>model_version, model_info]
    
    ReturnJSON --> DisplayResults[Display Results<br/>with Sentiment Icon]
    DisplayResults --> ShowModelUsed[Show Model Used<br/>Version1 or Version2]
    
    ShowModelUsed --> UserDecision{User Action?}
    UserDecision -->|Analyze New Text| EnterText
    UserDecision -->|Change Model| UserSelectModel
    UserDecision -->|Clear| DisabledState
    UserDecision -->|Done| End([End])

    style Start fill:#4CAF50,color:#fff
    style End fill:#f44336,color:#fff
    style UserSelectModel fill:#2196F3,color:#fff
    style LoadModel fill:#2196F3,color:#fff
    style Predict fill:#FF9800,color:#fff
    style DisplayResults fill:#4CAF50,color:#fff
```

## Training Flow - Model Creation

```mermaid
flowchart TD
    StartTrain([Training Process]) --> CheckVersion{Which Model<br/>to Train?}
    
    CheckVersion -->|Version 1| RunV1[Run train_cpu_model.py]
    CheckVersion -->|Version 2| SetupFirst{Version 1<br/>Exists?}
    
    SetupFirst -->|No| RunSetup[Run setup_model_versions.py<br/>Create Version 1 Structure]
    SetupFirst -->|Yes| RunV2[Run train_model_v2.py]
    RunSetup --> RunV2
    
    RunV1 --> LoadData1[Load Training Data<br/>social_media_sentiment_train.csv]
    RunV2 --> LoadData2[Load Training Data<br/>social_media_sentiment_train.csv]
    
    LoadData1 --> CreateToken1[Create Tokenizer<br/>Vocab: 500 words]
    LoadData2 --> CreateToken2[Create Tokenizer<br/>Vocab: 1000 words]
    
    CreateToken1 --> BuildModel1[Build Model V1<br/>1x Bi-LSTM Layer<br/>64 Embedding Dim]
    CreateToken2 --> BuildModel2[Build Model V2<br/>2x Bi-LSTM Layers<br/>128 Embedding Dim]
    
    BuildModel1 --> Train1[Train Model<br/>4 Epochs]
    BuildModel2 --> Train2[Train Model<br/>10 Epochs]
    
    Train1 --> SaveV1[Save to models/version1/<br/>- sentiment_model/1/<br/>- tokenizer.pickle<br/>- label_encoder.pickle<br/>- metadata.json]
    
    Train2 --> SaveV2[Save to models/version2/<br/>- sentiment_model/1/<br/>- tokenizer.pickle<br/>- label_encoder.pickle<br/>- metadata.json]
    
    SaveV1 --> TrainComplete([Training Complete])
    SaveV2 --> TrainComplete
    
    style StartTrain fill:#4CAF50,color:#fff
    style TrainComplete fill:#4CAF50,color:#fff
    style BuildModel1 fill:#2196F3,color:#fff
    style BuildModel2 fill:#FF9800,color:#fff
```

## Server Startup Flow

```mermaid
flowchart TD
    ServerStart([Start Server]) --> Activate[Activate Virtual Env<br/>sentiment_slang]
    
    Activate --> RunUvicorn[Run Command<br/>uvicorn app.main:app --reload]
    
    RunUvicorn --> ImportApp[Import FastAPI App<br/>app/main.py]
    
    ImportApp --> CheckMulti{utils_multimodel.py<br/>Available?}
    
    CheckMulti -->|Yes| LoadMulti[Load Multi-Model Utils<br/>MULTI_MODEL = True]
    CheckMulti -->|No| LoadSingle[Load Single Model Utils<br/>MULTI_MODEL = False]
    
    LoadMulti --> LoadAllModels[Load All Models<br/>Loop through AVAILABLE_MODELS]
    
    LoadAllModels --> LoadV1Model[Load Version 1<br/>TensorFlow SavedModel<br/>Tokenizer + Encoder]
    LoadV1Model --> LoadV2Model[Load Version 2<br/>TensorFlow SavedModel<br/>Tokenizer + Encoder]
    
    LoadV2Model --> LoadMetadata[Load Metadata JSON<br/>for Both Versions]
    
    LoadMetadata --> RegisterRoutes[Register API Routes<br/>GET /<br/>GET /models<br/>POST /predict]
    
    LoadSingle --> RegisterRoutes
    
    RegisterRoutes --> MountStatic[Mount Static Files<br/>/static -> app/static/]
    
    MountStatic --> ServerReady[Server Ready<br/>Listening on 127.0.0.1:8000]
    
    ServerReady --> WaitRequest[Wait for HTTP Requests]
    
    style ServerStart fill:#4CAF50,color:#fff
    style ServerReady fill:#4CAF50,color:#fff
    style LoadV1Model fill:#2196F3,color:#fff
    style LoadV2Model fill:#FF9800,color:#fff
```

## Data Preprocessing Flow

```mermaid
flowchart TD
    RawText([Raw Input Text]) --> RemoveURL[Remove URLs<br/>Regex Pattern]
    
    RemoveURL --> Lowercase[Convert to Lowercase]
    
    Lowercase --> DemojizeEmoji[Demojize Emojis<br/>😊 → :smiling_face:]
    
    DemojizeEmoji --> ExpandSlang[Expand Slang<br/>Dictionary Mapping<br/>lol → laughing out loud]
    
    ExpandSlang --> RemoveSpecial[Remove Special Characters<br/>Keep Only Letters/Spaces]
    
    RemoveSpecial --> StripWhitespace[Strip Extra Whitespace]
    
    StripWhitespace --> CleanedText([Cleaned Text])
    
    CleanedText --> ReturnAPI[Return to API]
    
    style RawText fill:#2196F3,color:#fff
    style CleanedText fill:#4CAF50,color:#fff
    style ExpandSlang fill:#FF9800,color:#fff
```

## Model Architecture Comparison

```mermaid
flowchart LR
    subgraph Version1[Version 1 - Baseline]
        V1Input[Input Text] --> V1Embed[Embedding Layer<br/>64 Dimensions<br/>500 Vocab]
        V1Embed --> V1LSTM[Bi-LSTM Layer<br/>64 Units]
        V1LSTM --> V1Drop[Dropout 0.5]
        V1Drop --> V1Dense1[Dense Layer<br/>32 Units]
        V1Dense1 --> V1Drop2[Dropout 0.3]
        V1Drop2 --> V1Output[Output<br/>4 Classes<br/>Softmax]
    end
    
    subgraph Version2[Version 2 - Enhanced]
        V2Input[Input Text] --> V2Embed[Embedding Layer<br/>128 Dimensions<br/>1000 Vocab]
        V2Embed --> V2LSTM1[Bi-LSTM Layer 1<br/>128 Units<br/>Return Sequences]
        V2LSTM1 --> V2Drop1[Dropout 0.3]
        V2Drop1 --> V2LSTM2[Bi-LSTM Layer 2<br/>64 Units]
        V2LSTM2 --> V2Drop2[Dropout 0.5]
        V2Drop2 --> V2Dense1[Dense Layer<br/>64 Units]
        V2Dense1 --> V2Drop3[Dropout 0.3]
        V2Drop3 --> V2Output[Output<br/>4 Classes<br/>Softmax]
    end
    
    style V1Output fill:#4CAF50,color:#fff
    style V2Output fill:#FF9800,color:#fff
```

## File Structure Flow

```mermaid
graph TD
    Root[sentiment_slang_ca1/] --> App[app/]
    Root --> Data[data/]
    Root --> Models[models/]
    Root --> VEnv[sentiment_slang/]
    Root --> Files[Config Files]
    
    App --> AppMain[main.py<br/>FastAPI Routes]
    App --> AppUtils[utils_multimodel.py<br/>Model Loader]
    App --> AppPrep[preprocessing.py<br/>Text Cleaning]
    App --> AppStatic[static/]
    
    AppStatic --> HTML[index.html<br/>Web Interface]
    AppStatic --> CSS[style.css<br/>Styling]
    AppStatic --> JS[script.js<br/>Frontend Logic]
    
    Data --> TrainCSV[social_media_sentiment_train.csv]
    Data --> TestCSV[social_media_sentiment_test.csv]
    
    Models --> V1[version1/]
    Models --> V2[version2/]
    
    V1 --> V1Model[sentiment_model/1/<br/>SavedModel Format]
    V1 --> V1Token[tokenizer.pickle]
    V1 --> V1Encode[label_encoder.pickle]
    V1 --> V1Meta[metadata.json]
    
    V2 --> V2Model[sentiment_model/1/<br/>SavedModel Format]
    V2 --> V2Token[tokenizer.pickle]
    V2 --> V2Encode[label_encoder.pickle]
    V2 --> V2Meta[metadata.json]
    
    Files --> Docker[dockerfile<br/>docker-compose.yml]
    Files --> Req[requirements.txt]
    Files --> Train[train_cpu_model.py<br/>train_model_v2.py]
    Files --> Setup[setup_model_versions.py]
    
    style AppMain fill:#2196F3,color:#fff
    style AppUtils fill:#FF9800,color:#fff
    style V1Model fill:#4CAF50,color:#fff
    style V2Model fill:#4CAF50,color:#fff
```

---

## How to View the Flowcharts

### Option 1: VS Code (Recommended)
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file in VS Code
3. Press `Ctrl+Shift+V` to preview
4. Flowcharts will render as images

### Option 2: Online Mermaid Editor
1. Copy any diagram code (between ```mermaid and ```)
2. Go to https://mermaid.live/
3. Paste and see the rendered flowchart
4. Export as PNG/SVG

### Option 3: GitHub/GitLab
- Upload this file to GitHub/GitLab
- Mermaid diagrams render automatically

---

## Legend

- 🟢 Green: Start/End/Success states
- 🔵 Blue: Decision points/User interactions
- 🟠 Orange: Processing/Model operations
- ⬜ White: Regular flow steps

---

**Generated:** March 6, 2026
**Project:** Sentiment Slang Analyzer - Multi-Model System
