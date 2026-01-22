#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QTextEdit>
#include <QFileDialog>
#include <QLabel>
#include <QProgressBar>
#include <QMessageBox>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QSettings>
#include <QTabWidget>
#include <QPixmap>
#include <QScrollArea>
#include <QScrollBar>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <QRegularExpression>
#include <QJsonDocument>
#include <QJsonObject>
#include <QQueue>
#include <fstream>
#include <utility>

#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

enum SDMode { TXT2IMG, IMG2IMG, CONVERT };

struct SDParams {
    SDMode mode = TXT2IMG;
    std::string model_path;
    std::string vae_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string output_path = "output.png";
    std::string preview_path;
    std::string input_path;
    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    float guidance = 3.5f;
    int width = 512;
    int height = 512;
    int sample_steps = 20;
    float strength = 0.75f;
    sample_method_t sample_method = EULER_A_SAMPLE_METHOD;
    int64_t seed = 42;
    int n_threads = -1;
    bool verbose = false;
    bool clip_on_cpu = false;
};

class GenerationWorker : public QThread {
    Q_OBJECT

public:
    GenerationWorker(const SDParams& params) : params_(params) {}

signals:
    void finished(bool success, const QString& message, const QString& imagePath, const QString& arguments);

protected:
    void run() override {
        QString args = formatArguments();
        
        if (params_.mode == CONVERT) {
            bool success = convert(params_.model_path.c_str(), params_.vae_path.c_str(), 
                                 params_.output_path.c_str(), SD_TYPE_COUNT, nullptr, false);
            emit finished(success, success ? "Conversion completed" : "Conversion failed", 
                         success ? QString::fromStdString(params_.output_path) : "", args);
            return;
        }

        sd_ctx_params_t ctx_params;
        sd_ctx_params_init(&ctx_params);
        ctx_params.model_path = params_.model_path.c_str();
        // Only set VAE if specified and is a GGUF file, otherwise use embedded VAE
        ctx_params.vae_path = (params_.vae_path.empty() ||
                               (!params_.vae_path.empty() && params_.vae_path.find(".gguf") == std::string::npos))
                               ? nullptr : params_.vae_path.c_str();
        ctx_params.clip_l_path = params_.clip_l_path.empty() ? nullptr : params_.clip_l_path.c_str();
        ctx_params.clip_g_path = params_.clip_g_path.empty() ? nullptr : params_.clip_g_path.c_str();
        ctx_params.t5xxl_path = params_.t5xxl_path.empty() ? nullptr : params_.t5xxl_path.c_str();
        // Set threads to auto-detect if not specified (matches CLI behavior)
        ctx_params.n_threads = (params_.n_threads <= 0) ? sd_get_num_physical_cores() : params_.n_threads;
        ctx_params.wtype = SD_TYPE_COUNT;
        ctx_params.rng_type = CUDA_RNG;
        ctx_params.prediction = PREDICTION_COUNT;  // Auto-detect prediction type
        ctx_params.vae_decode_only = false;  // CRITICAL: Must be false for image generation
        ctx_params.free_params_immediately = true;
        ctx_params.keep_clip_on_cpu = params_.clip_on_cpu;
        
        // Log initialization parameters for debugging
        QString debugMsg = QString("Initializing SD context:\n"
                                   "Model: %1\n"
                                   "VAE: %2\n"
                                   "CLIP-L: %3\n"
                                   "CLIP-G: %4\n"
                                   "T5XXL: %5\n"
                                   "vae_decode_only: %6\n"
                                   "prediction: %7\n"
                                   "threads: %8")
            .arg(ctx_params.model_path ? ctx_params.model_path : "null")
            .arg(ctx_params.vae_path ? ctx_params.vae_path : "null")
            .arg(ctx_params.clip_l_path ? ctx_params.clip_l_path : "null")
            .arg(ctx_params.clip_g_path ? ctx_params.clip_g_path : "null")
            .arg(ctx_params.t5xxl_path ? ctx_params.t5xxl_path : "null")
            .arg(ctx_params.vae_decode_only)
            .arg(ctx_params.prediction)
            .arg(ctx_params.n_threads);
        qDebug() << debugMsg;
        fprintf(stdout, "[DEBUG] %s\n", debugMsg.toStdString().c_str());
        fflush(stdout);

        sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);

        if (!sd_ctx) {
            QString errorMsg = "Failed to initialize SD context.\n\nDebug info:\n" + debugMsg;
            emit finished(false, errorMsg, "", args);
            return;
        }

        // Setup preview callback if preview path is provided
        if (!params_.preview_path.empty()) {
            auto preview_callback = [](int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
                if (frame_count > 0 && frames && frames[0].data) {
                    std::string* preview_path = static_cast<std::string*>(data);
                    stbi_write_png(preview_path->c_str(), frames[0].width, frames[0].height,
                                 frames[0].channel, frames[0].data, 0, nullptr);
                }
            };
            sd_set_preview_callback(preview_callback, PREVIEW_VAE, 1, true, false,
                                   const_cast<void*>(static_cast<const void*>(&params_.preview_path)));
        }

        sd_image_t* results = nullptr;
        
        if (params_.mode == TXT2IMG) {
            sd_img_gen_params_t gen_params;
            sd_img_gen_params_init(&gen_params);
            gen_params.prompt = params_.prompt.c_str();
            gen_params.negative_prompt = params_.negative_prompt.c_str();
            gen_params.width = params_.width;
            gen_params.height = params_.height;
            gen_params.seed = params_.seed;
            gen_params.batch_count = 1;
            gen_params.sample_params.sample_method = params_.sample_method;
            gen_params.sample_params.sample_steps = params_.sample_steps;
            gen_params.sample_params.guidance.txt_cfg = params_.cfg_scale;
            gen_params.sample_params.guidance.img_cfg = params_.guidance;
            
            results = generate_image(sd_ctx, &gen_params);
        } else if (params_.mode == IMG2IMG) {
            int c = 0, w = 0, h = 0;
            uint8_t* input_buffer = stbi_load(params_.input_path.c_str(), &w, &h, &c, 3);
            if (!input_buffer) {
                free_sd_ctx(sd_ctx);
                emit finished(false, "Failed to load input image", "", args);
                return;
            }

            if (w != params_.width || h != params_.height) {
                uint8_t* resized = (uint8_t*)malloc(params_.width * params_.height * 3);
                stbir_resize(input_buffer, w, h, 0, resized, params_.width, params_.height, 0,
                           STBIR_TYPE_UINT8, 3, STBIR_ALPHA_CHANNEL_NONE, 0,
                           STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                           STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                           STBIR_COLORSPACE_SRGB, nullptr);
                free(input_buffer);
                input_buffer = resized;
            }

            sd_image_t input_image = {(uint32_t)params_.width, (uint32_t)params_.height, 3, input_buffer};
            std::vector<uint8_t> mask_data(params_.width * params_.height, 255);
            sd_image_t mask_image = {(uint32_t)params_.width, (uint32_t)params_.height, 1, mask_data.data()};

            sd_img_gen_params_t gen_params;
            sd_img_gen_params_init(&gen_params);
            gen_params.prompt = params_.prompt.c_str();
            gen_params.negative_prompt = params_.negative_prompt.c_str();
            gen_params.init_image = input_image;
            gen_params.mask_image = mask_image;
            gen_params.width = params_.width;
            gen_params.height = params_.height;
            gen_params.seed = params_.seed;
            gen_params.batch_count = 1;
            gen_params.strength = params_.strength;
            gen_params.sample_params.sample_method = params_.sample_method;
            gen_params.sample_params.sample_steps = params_.sample_steps;
            gen_params.sample_params.guidance.txt_cfg = params_.cfg_scale;
            gen_params.sample_params.guidance.img_cfg = params_.guidance;
            
            results = generate_image(sd_ctx, &gen_params);
            free(input_buffer);
        }

        bool success = false;
        if (results && results->data) {
            success = stbi_write_png(params_.output_path.c_str(), results->width,
                                   results->height, results->channel,
                                   results->data, 0, nullptr);

            if (success) {
                std::string txt_path = params_.output_path;
                size_t dot_pos = txt_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    txt_path = txt_path.substr(0, dot_pos) + ".txt";
                } else {
                    txt_path += ".txt";
                }

                std::ofstream txt_file(txt_path);
                if (txt_file.is_open()) {
                    txt_file << args.toStdString();
                    txt_file.close();
                }
            }

            free(results->data);
        }
        
        if (results) free(results);
        free_sd_ctx(sd_ctx);
        
        emit finished(success, success ? "Generation completed" : "Generation failed",
                     success ? QString::fromStdString(params_.output_path) : "", args);
    }

private:
    QString formatArguments() {
        QJsonObject json;
        json["mode"] = params_.mode == TXT2IMG ? "txt2img" : 
                       params_.mode == IMG2IMG ? "img2img" : "convert";
        json["model"] = QString::fromStdString(params_.model_path);
        if (!params_.vae_path.empty())
            json["vae"] = QString::fromStdString(params_.vae_path);
        if (!params_.clip_l_path.empty())
            json["clip_l"] = QString::fromStdString(params_.clip_l_path);
        if (!params_.clip_g_path.empty())
            json["clip_g"] = QString::fromStdString(params_.clip_g_path);
        if (!params_.t5xxl_path.empty())
            json["t5xxl"] = QString::fromStdString(params_.t5xxl_path);
        if (!params_.preview_path.empty())
            json["preview_path"] = QString::fromStdString(params_.preview_path);
        if (params_.clip_on_cpu)
            json["clip_on_cpu"] = true;
        if (params_.mode != CONVERT) {
            json["prompt"] = QString::fromStdString(params_.prompt);
            if (!params_.negative_prompt.empty())
                json["negative_prompt"] = QString::fromStdString(params_.negative_prompt);
            json["cfg_scale"] = params_.cfg_scale;
            json["width"] = params_.width;
            json["height"] = params_.height;
            json["steps"] = params_.sample_steps;
            json["seed"] = params_.seed;
        }
        return QJsonDocument(json).toJson(QJsonDocument::Indented);
    }
    
    SDParams params_;
};

class MainWindow : public QWidget {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr) : QWidget(parent) {
        setupUI();
        connectSignals();
        loadSettings();

        // Setup preview timer
        previewTimer_ = new QTimer(this);
        connect(previewTimer_, &QTimer::timeout, this, &MainWindow::updatePreview);

        appendLog("[INFO] Stable Diffusion Qt6 initialized.");
    }

private slots:
    void browseModel() {
        QString file = QFileDialog::getOpenFileName(this, "Select Model", "", "Model Files (*.gguf *.safetensors)");
        if (!file.isEmpty()) modelPath_->setText(file);
    }

    void browseVAE() {
        QString file = QFileDialog::getOpenFileName(this, "Select VAE", "", "VAE Files (*.gguf *.safetensors)");
        if (!file.isEmpty()) vaePath_->setText(file);
    }

    void browseClipL() {
        QString file = QFileDialog::getOpenFileName(this, "Select CLIP-L", "", "CLIP Files (*.safetensors)");
        if (!file.isEmpty()) clipLPath_->setText(file);
    }

    void browseClipG() {
        QString file = QFileDialog::getOpenFileName(this, "Select CLIP-G", "", "CLIP Files (*.safetensors)");
        if (!file.isEmpty()) clipGPath_->setText(file);
    }

    void browseT5XXL() {
        QString file = QFileDialog::getOpenFileName(this, "Select T5-XXL", "", "T5 Files (*.safetensors)");
        if (!file.isEmpty()) t5xxlPath_->setText(file);
    }

    void browseInput() {
        QString file = QFileDialog::getOpenFileName(this, "Select Input Image", "", "Images (*.png *.jpg *.jpeg)");
        if (!file.isEmpty()) inputPath_->setText(file);
    }

    void browseOutput() {
        QString file = QFileDialog::getSaveFileName(this, "Save Output", "output.png", "Images (*.png *.jpg)");
        if (!file.isEmpty()) outputPath_->setText(file);
    }

    void onModeChanged() {
        bool isImg2Img = modeCombo_->currentIndex() == 1;
        inputPath_->setEnabled(isImg2Img);
        browseInputBtn_->setEnabled(isImg2Img);
        strength_->setEnabled(isImg2Img);
        
        bool isConvert = modeCombo_->currentIndex() == 2;
        prompt_->setEnabled(!isConvert);
        negativePrompt_->setEnabled(!isConvert);
        cfgScale_->setEnabled(!isConvert);
        guidance_->setEnabled(!isConvert);
        width_->setEnabled(!isConvert);
        height_->setEnabled(!isConvert);
        steps_->setEnabled(!isConvert);
        seed_->setEnabled(!isConvert);
        samplerCombo_->setEnabled(!isConvert);
    }

    void generate() {
        if (modelPath_->text().isEmpty()) {
            QMessageBox::warning(this, "Error", "Please select a model file");
            return;
        }

        saveSettings();

        SDParams params;
        params.mode = (SDMode)modeCombo_->currentIndex();

        // Store QString conversions in variables to ensure proper lifetime
        std::string model = modelPath_->text().toStdString();
        std::string vae = vaePath_->text().toStdString();
        std::string clip_l = clipLPath_->text().toStdString();
        std::string clip_g = clipGPath_->text().toStdString();
        std::string t5xxl = t5xxlPath_->text().toStdString();
        std::string output = outputPath_->text().toStdString();
        std::string input = inputPath_->text().toStdString();
        std::string prompt = prompt_->toPlainText().toStdString();
        std::string neg_prompt = negativePrompt_->toPlainText().toStdString();

        // Generate unique preview path in temp directory
        QString tempDir = QDir::tempPath();
        QString previewFileName = QString("sd_preview_%1.png").arg(QDateTime::currentMSecsSinceEpoch());
        std::string preview = QDir(tempDir).filePath(previewFileName).toStdString();

        params.model_path = std::move(model);
        params.vae_path = std::move(vae);
        params.clip_l_path = std::move(clip_l);
        params.clip_g_path = std::move(clip_g);
        params.t5xxl_path = std::move(t5xxl);
        params.output_path = std::move(output);
        params.preview_path = std::move(preview);
        params.input_path = std::move(input);
        params.prompt = std::move(prompt);
        params.negative_prompt = std::move(neg_prompt);
        params.cfg_scale = cfgScale_->value();
        params.guidance = guidance_->value();
        params.width = width_->value();
        params.height = height_->value();
        params.sample_steps = steps_->value();
        params.strength = strength_->value();
        params.sample_method = (sample_method_t)samplerCombo_->currentIndex();
        params.seed = seed_->value();
        params.n_threads = threads_->value();
        params.verbose = verbose_->isChecked();
        params.clip_on_cpu = clipOnCpu_->isChecked();

        jobQueue_.enqueue(params);
        updateJobsProgress();
        processNextJob();
    }

    void onGenerationFinished(bool success, const QString& message, const QString& imagePath, const QString& arguments) {
        completedJobs_++;
        updateJobsProgress();
        previewTimer_->stop();

        // Remove preview tab
        if (previewTabIndex_ >= 0 && previewTabIndex_ < tabWidget_->count()) {
            tabWidget_->removeTab(previewTabIndex_);
        }
        previewTabIndex_ = -1;

        if (success) {
            addResultTab(imagePath, arguments);
            saveImageToSettings(imagePath, arguments);
            setWindowTitle(QString("Stable Diffusion Qt6 - %1").arg(message));
            appendLog(QString("[SUCCESS] %1 - Image saved to: %2").arg(message).arg(imagePath));
        } else {
            addResultTab("", arguments, true, message);
            //QMessageBox::warning(this, "Generation Error", message);
            setWindowTitle("Stable Diffusion Qt6 - Generation Failed");
            appendLog(QString("[ERROR] %1").arg(message));
        }

        currentWorker_->deleteLater();
        currentWorker_ = nullptr;
        processNextJob();
    }
    
    void processNextJob() {
        if (currentWorker_ != nullptr) {
            return;
        }

        if (jobQueue_.isEmpty()) {
            progressBar_->setVisible(false);
            jobsProgressLabel_->setVisible(false);
            previewTimer_->stop();
            currentPreviewPath_.clear();
            previewTabIndex_ = -1;
            appendLog("[INFO] All jobs completed.");
            return;
        }

        progressBar_->setVisible(true);
        jobsProgressLabel_->setVisible(true);

        SDParams params = jobQueue_.dequeue();
        currentPreviewPath_ = QString::fromStdString(params.preview_path);

        appendLog(QString("[INFO] Starting generation: %1x%2, steps: %3, seed: %4")
                  .arg(params.width).arg(params.height).arg(params.sample_steps).arg(params.seed));
        appendLog(QString("[INFO] Preview path: %1").arg(currentPreviewPath_));

        // Create preview tab
        auto* previewTabWidget = new QWidget;
        auto* previewLayout = new QVBoxLayout(previewTabWidget);

        auto* imageLabel = new QLabel;
        imageLabel->setText("Generating...");
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setMinimumSize(512, 512);

        auto* scrollArea = new QScrollArea;
        scrollArea->setWidget(imageLabel);
        scrollArea->setWidgetResizable(true);
        previewLayout->addWidget(scrollArea);

        previewTabIndex_ = tabWidget_->addTab(previewTabWidget, "⏳ Generating...");
        tabWidget_->setCurrentIndex(previewTabIndex_);

        currentWorker_ = new GenerationWorker(params);
        connect(currentWorker_, &GenerationWorker::finished, this, &MainWindow::onGenerationFinished);
        currentWorker_->start();

        // Start preview timer to update every 500ms
        previewTimer_->start(500);
    }
    
    void updateJobsProgress() {
        int total = completedJobs_ + jobQueue_.size() + (currentWorker_ ? 1 : 0);
        if (total == 0) total = 1;
        jobsProgressLabel_->setText(QString("Completed Jobs: %1/%2").arg(completedJobs_).arg(total));
        if (completedJobs_ == total) {
            completedJobs_ = 0;
        }
    }
    


private:
    void setupUI() {
        setWindowTitle("Stable Diffusion Qt6");
        setMinimumSize(1200, 800);

        auto* mainLayout = new QHBoxLayout(this);
        
        // Left side - controls
        auto* leftWidget = new QWidget;
        leftWidget->setMaximumWidth(400);
        auto* layout = new QVBoxLayout(leftWidget);
        auto* formLayout = new QFormLayout;

        // Mode selection
        modeCombo_ = new QComboBox;
        modeCombo_->addItems({"txt2img", "img2img", "convert"});
        formLayout->addRow("Mode:", modeCombo_);

        // Model path
        auto* modelLayout = new QHBoxLayout;
        modelPath_ = new QLineEdit;
        browseModelBtn_ = new QPushButton("Browse");
        modelLayout->addWidget(modelPath_);
        modelLayout->addWidget(browseModelBtn_);
        formLayout->addRow("Model:", modelLayout);

        // VAE path
        auto* vaeLayout = new QHBoxLayout;
        vaePath_ = new QLineEdit;
        browseVAEBtn_ = new QPushButton("Browse");
        vaeLayout->addWidget(vaePath_);
        vaeLayout->addWidget(browseVAEBtn_);
        formLayout->addRow("VAE:", vaeLayout);

        // CLIP-L path
        auto* clipLLayout = new QHBoxLayout;
        clipLPath_ = new QLineEdit;
        browseClipLBtn_ = new QPushButton("Browse");
        clipLLayout->addWidget(clipLPath_);
        clipLLayout->addWidget(browseClipLBtn_);
        formLayout->addRow("CLIP-L:", clipLLayout);

        // CLIP-G path
        auto* clipGLayout = new QHBoxLayout;
        clipGPath_ = new QLineEdit;
        browseClipGBtn_ = new QPushButton("Browse");
        clipGLayout->addWidget(clipGPath_);
        clipGLayout->addWidget(browseClipGBtn_);
        formLayout->addRow("CLIP-G:", clipGLayout);

        // T5-XXL path
        auto* t5xxlLayout = new QHBoxLayout;
        t5xxlPath_ = new QLineEdit;
        browseT5XXLBtn_ = new QPushButton("Browse");
        t5xxlLayout->addWidget(t5xxlPath_);
        t5xxlLayout->addWidget(browseT5XXLBtn_);
        formLayout->addRow("T5-XXL:", t5xxlLayout);

        // Input image
        auto* inputLayout = new QHBoxLayout;
        inputPath_ = new QLineEdit;
        inputPath_->setEnabled(false);
        browseInputBtn_ = new QPushButton("Browse");
        browseInputBtn_->setEnabled(false);
        inputLayout->addWidget(inputPath_);
        inputLayout->addWidget(browseInputBtn_);
        formLayout->addRow("Input Image:", inputLayout);

        // Output path
        auto* outputLayout = new QHBoxLayout;
        outputPath_ = new QLineEdit("output.png");
        browseOutputBtn_ = new QPushButton("Browse");
        outputLayout->addWidget(outputPath_);
        outputLayout->addWidget(browseOutputBtn_);
        formLayout->addRow("Output:", outputLayout);

        // Prompt
        prompt_ = new QTextEdit;
        prompt_->setMaximumHeight(80);
        formLayout->addRow("Prompt:", prompt_);

        // Negative prompt
        negativePrompt_ = new QTextEdit;
        negativePrompt_->setMaximumHeight(60);
        formLayout->addRow("Negative Prompt:", negativePrompt_);

        // Parameters
        cfgScale_ = new QDoubleSpinBox;
        cfgScale_->setRange(1.0, 30.0);
        cfgScale_->setValue(7.0);
        cfgScale_->setSingleStep(0.5);
        formLayout->addRow("CFG Scale:", cfgScale_);

        guidance_ = new QDoubleSpinBox;
        guidance_->setRange(1.0, 20.0);
        guidance_->setValue(3.5);
        guidance_->setSingleStep(0.5);
        formLayout->addRow("Guidance:", guidance_);

        width_ = new QSpinBox;
        width_->setRange(64, 2048);
        width_->setValue(512);
        width_->setSingleStep(64);
        formLayout->addRow("Width:", width_);

        height_ = new QSpinBox;
        height_->setRange(64, 2048);
        height_->setValue(512);
        height_->setSingleStep(64);
        formLayout->addRow("Height:", height_);

        steps_ = new QSpinBox;
        steps_->setRange(1, 100);
        steps_->setValue(20);
        formLayout->addRow("Steps:", steps_);

        strength_ = new QDoubleSpinBox;
        strength_->setRange(0.0, 1.0);
        strength_->setValue(0.75);
        strength_->setSingleStep(0.05);
        strength_->setEnabled(false);
        formLayout->addRow("Strength:", strength_);

        samplerCombo_ = new QComboBox;
        samplerCombo_->addItems({"euler_a", "euler", "heun", "dpm2", "dpm++2s_a", "dpm++2m"});
        formLayout->addRow("Sampler:", samplerCombo_);

        seed_ = new QSpinBox;
        seed_->setRange(-1, INT_MAX);
        seed_->setValue(42);
        formLayout->addRow("Seed:", seed_);

        threads_ = new QSpinBox;
        threads_->setRange(-1, 32);
        threads_->setValue(-1);
        formLayout->addRow("Threads:", threads_);

        verbose_ = new QCheckBox;
        formLayout->addRow("Verbose:", verbose_);

        clipOnCpu_ = new QCheckBox;
        formLayout->addRow("CLIP on CPU:", clipOnCpu_);

        layout->addLayout(formLayout);

        // Generate button
        generateBtn_ = new QPushButton("Generate");
        generateBtn_->setMinimumHeight(40);
        layout->addWidget(generateBtn_);

        // Progress bar
        progressBar_ = new QProgressBar;
        progressBar_->setRange(0, 0);
        progressBar_->setVisible(false);
        layout->addWidget(progressBar_);
        
        // Jobs progress
        jobsProgressLabel_ = new QLabel("Completed Jobs: 0/0");
        jobsProgressLabel_->setVisible(false);
        layout->addWidget(jobsProgressLabel_);
        
        mainLayout->addWidget(leftWidget);

        // Right side - vertical split with tabs and log
        auto* rightWidget = new QWidget;
        auto* rightLayout = new QVBoxLayout(rightWidget);
        rightLayout->setContentsMargins(0, 0, 0, 0);

        // Results tabs
        tabWidget_ = new QTabWidget;
        tabWidget_->setTabsClosable(true);
        connect(tabWidget_, &QTabWidget::tabCloseRequested, this, &MainWindow::closeTab);
        rightLayout->addWidget(tabWidget_, 3);

        // Log window
        auto* logLabel = new QLabel("Log:");
        rightLayout->addWidget(logLabel);

        logText_ = new QTextEdit;
        logText_->setReadOnly(true);
        logText_->setMaximumHeight(150);
        logText_->setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; background-color: #2b2b2b; color: #d4d4d4; }");
        rightLayout->addWidget(logText_);

        mainLayout->addWidget(rightWidget);
    }
    
    void addResultTab(const QString& imagePath, const QString& arguments, bool isError = false, const QString& errorMessage = "") {
        auto* tabWidget = new QWidget;
        auto* tabLayout = new QVBoxLayout(tabWidget);

        if (isError) {
            // Error message display (selectable text)
            auto* errorText = new QTextEdit;
            errorText->setPlainText(errorMessage);
            errorText->setReadOnly(true);
            errorText->setStyleSheet("QTextEdit { padding: 10px; background-color: #ffe6e6; border: 2px solid #ff4444; color: #cc0000; font-family: monospace; }");
            errorText->setMinimumHeight(200);
            tabLayout->addWidget(errorText);
        } else {
            // Image display
            auto* imageLabel = new QLabel;
            QPixmap pixmap(imagePath);
            if (!pixmap.isNull()) {
                imageLabel->setPixmap(pixmap.scaled(512, 512, Qt::KeepAspectRatio, Qt::SmoothTransformation));
            } else {
                imageLabel->setText("Image not found");
            }
            imageLabel->setAlignment(Qt::AlignCenter);

            auto* scrollArea = new QScrollArea;
            scrollArea->setWidget(imageLabel);
            scrollArea->setWidgetResizable(true);
            tabLayout->addWidget(scrollArea);
        }

        // Arguments display
        auto* argsText = new QTextEdit;
        argsText->setPlainText(arguments);
        argsText->setMaximumHeight(150);
        argsText->setReadOnly(true);
        tabLayout->addWidget(argsText);

        QString tabName = isError ? "Error" : QFileInfo(imagePath).fileName();
        int tabIndex = tabWidget_->addTab(tabWidget, tabName);
        tabWidget_->setTabToolTip(tabIndex, isError ? errorMessage : imagePath);
        tabWidget_->setCurrentWidget(tabWidget);
    }
    
    void closeTab(int index) {
        QString imagePath = tabWidget_->tabToolTip(index);
        if (!imagePath.isEmpty()) {
            removeImageFromSettings(imagePath);
        }
        tabWidget_->removeTab(index);
    }

    void connectSignals() {
        connect(browseModelBtn_, &QPushButton::clicked, this, &MainWindow::browseModel);
        connect(browseVAEBtn_, &QPushButton::clicked, this, &MainWindow::browseVAE);
        connect(browseClipLBtn_, &QPushButton::clicked, this, &MainWindow::browseClipL);
        connect(browseClipGBtn_, &QPushButton::clicked, this, &MainWindow::browseClipG);
        connect(browseT5XXLBtn_, &QPushButton::clicked, this, &MainWindow::browseT5XXL);
        connect(browseInputBtn_, &QPushButton::clicked, this, &MainWindow::browseInput);
        connect(browseOutputBtn_, &QPushButton::clicked, this, &MainWindow::browseOutput);
        connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onModeChanged);
        connect(generateBtn_, &QPushButton::clicked, this, &MainWindow::generate);
    }

    void saveSettings() {
        QSettings settings;
        settings.setValue("mode", modeCombo_->currentIndex());
        settings.setValue("modelPath", modelPath_->text());
        settings.setValue("vaePath", vaePath_->text());
        settings.setValue("clipLPath", clipLPath_->text());
        settings.setValue("clipGPath", clipGPath_->text());
        settings.setValue("t5xxlPath", t5xxlPath_->text());
        settings.setValue("inputPath", inputPath_->text());
        settings.setValue("outputPath", outputPath_->text());
        settings.setValue("prompt", prompt_->toPlainText());
        settings.setValue("negativePrompt", negativePrompt_->toPlainText());
        settings.setValue("cfgScale", cfgScale_->value());
        settings.setValue("guidance", guidance_->value());
        settings.setValue("width", width_->value());
        settings.setValue("height", height_->value());
        settings.setValue("steps", steps_->value());
        settings.setValue("strength", strength_->value());
        settings.setValue("sampler", samplerCombo_->currentIndex());
        settings.setValue("seed", seed_->value());
        settings.setValue("threads", threads_->value());
        settings.setValue("verbose", verbose_->isChecked());
        settings.setValue("clipOnCpu", clipOnCpu_->isChecked());
    }

    void loadSettings() {
        QSettings settings;
        qDebug() << "Loading from:" << settings.fileName();
        modeCombo_->setCurrentIndex(settings.value("mode", 0).toInt());
        modelPath_->setText(settings.value("modelPath").toString());
        vaePath_->setText(settings.value("vaePath").toString());
        clipLPath_->setText(settings.value("clipLPath").toString());
        clipGPath_->setText(settings.value("clipGPath").toString());
        t5xxlPath_->setText(settings.value("t5xxlPath").toString());
        inputPath_->setText(settings.value("inputPath").toString());
        outputPath_->setText(settings.value("outputPath", "output.png").toString());
        prompt_->setPlainText(settings.value("prompt").toString());
        negativePrompt_->setPlainText(settings.value("negativePrompt").toString());
        cfgScale_->setValue(settings.value("cfgScale", 7.0).toDouble());
        guidance_->setValue(settings.value("guidance", 3.5).toDouble());
        width_->setValue(settings.value("width", 512).toInt());
        height_->setValue(settings.value("height", 512).toInt());
        steps_->setValue(settings.value("steps", 20).toInt());
        strength_->setValue(settings.value("strength", 0.75).toDouble());
        samplerCombo_->setCurrentIndex(settings.value("sampler", 0).toInt());
        seed_->setValue(settings.value("seed", 42).toInt());
        threads_->setValue(settings.value("threads", -1).toInt());
        verbose_->setChecked(settings.value("verbose", false).toBool());
        clipOnCpu_->setChecked(settings.value("clipOnCpu", false).toBool());
        onModeChanged();
        loadPersistedImages();
    }

    // UI elements
    QComboBox* modeCombo_;
    QLineEdit* modelPath_;
    QLineEdit* vaePath_;
    QLineEdit* clipLPath_;
    QLineEdit* clipGPath_;
    QLineEdit* t5xxlPath_;
    QLineEdit* inputPath_;
    QLineEdit* outputPath_;
    QPushButton* browseModelBtn_;
    QPushButton* browseVAEBtn_;
    QPushButton* browseClipLBtn_;
    QPushButton* browseClipGBtn_;
    QPushButton* browseT5XXLBtn_;
    QPushButton* browseInputBtn_;
    QPushButton* browseOutputBtn_;
    QTextEdit* prompt_;
    QTextEdit* negativePrompt_;
    QDoubleSpinBox* cfgScale_;
    QDoubleSpinBox* guidance_;
    QSpinBox* width_;
    QSpinBox* height_;
    QSpinBox* steps_;
    QDoubleSpinBox* strength_;
    QComboBox* samplerCombo_;
    QSpinBox* seed_;
    QSpinBox* threads_;
    QCheckBox* verbose_;
    QCheckBox* clipOnCpu_;
    QPushButton* generateBtn_;
    QProgressBar* progressBar_;
    QTabWidget* tabWidget_;
    QLabel* jobsProgressLabel_;
    QTextEdit* logText_;
    QTimer* previewTimer_;

    QQueue<SDParams> jobQueue_;
    int completedJobs_ = 0;
    GenerationWorker* currentWorker_ = nullptr;
    QString currentPreviewPath_;
    int previewTabIndex_ = -1;

    void appendLog(const QString& message) {
        logText_->append(message);
        logText_->verticalScrollBar()->setValue(logText_->verticalScrollBar()->maximum());
    }

    void updatePreview() {
        if (currentPreviewPath_.isEmpty() || previewTabIndex_ < 0) {
            return;
        }

        QPixmap preview(currentPreviewPath_);
        if (!preview.isNull()) {
            // Update the preview tab's image
            QWidget* tabContent = tabWidget_->widget(previewTabIndex_);
            if (tabContent) {
                QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(tabContent->layout());
                if (layout && layout->count() > 0) {
                    QScrollArea* scrollArea = qobject_cast<QScrollArea*>(layout->itemAt(0)->widget());
                    if (scrollArea) {
                        QLabel* imageLabel = qobject_cast<QLabel*>(scrollArea->widget());
                        if (imageLabel) {
                            imageLabel->setPixmap(preview.scaled(512, 512, Qt::KeepAspectRatio, Qt::SmoothTransformation));
                        }
                    }
                }
            }
        }
    }
    
    void saveImageToSettings(const QString& imagePath, const QString& arguments) {
        qDebug() << "saveImageToSettings: img:" << imagePath << " ;args=" << arguments;
        QSettings settings;
        settings.beginGroup("GeneratedImages");
        QString key = QFileInfo(imagePath).fileName();
        QJsonDocument doc = QJsonDocument::fromJson(arguments.toUtf8());
        QJsonObject obj = doc.object();
        obj["output_path"] = imagePath;
        settings.setValue(key, QJsonDocument(obj).toJson(QJsonDocument::Compact));
        settings.endGroup();
    }
    
    void removeImageFromSettings(const QString& imagePath) {
        qDebug() << "removeImageFromSettings:" << imagePath;
        QSettings settings;
        settings.beginGroup("GeneratedImages");
        QString key = QFileInfo(imagePath).fileName();
        settings.remove(key);
        settings.endGroup();
    }
    
    void loadPersistedImages() {
        qDebug() << "loadPersistedImages ##################";
        QSettings settings;
        settings.beginGroup("GeneratedImages");
        QStringList keys = settings.allKeys();
        for (const QString& key : keys) {
            QString arguments = settings.value(key).toString();
            QJsonDocument doc = QJsonDocument::fromJson(arguments.toUtf8());
            QString imagePath = doc.object().value("output_path").toString();
            qDebug() << "loadPersistedImages key:" << key << "path:" << imagePath << "args:" << arguments;
            if (!imagePath.isEmpty() && QFile::exists(imagePath)) {
                addResultTab(imagePath, arguments);
            } else {
                settings.remove(key);
            }
        }
        settings.endGroup();
    }
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("StableDiffusionQt");
    app.setOrganizationName("StableDiffusionQt");
    
    MainWindow window;
    window.show();
    
    return app.exec();
}

#include "main.moc"