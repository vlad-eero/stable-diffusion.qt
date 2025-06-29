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
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QJsonDocument>
#include <QJsonObject>
#include <QQueue>
#include <fstream>

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
    std::string output_path = "output.png";
    std::string input_path;
    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    float guidance = 3.5f;
    int width = 512;
    int height = 512;
    int sample_steps = 20;
    float strength = 0.75f;
    sample_method_t sample_method = EULER_A;
    int64_t seed = 42;
    int n_threads = -1;
    bool verbose = false;
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
                                 params_.output_path.c_str(), SD_TYPE_COUNT);
            emit finished(success, success ? "Conversion completed" : "Conversion failed", 
                         success ? QString::fromStdString(params_.output_path) : "", args);
            return;
        }

        sd_ctx_t* sd_ctx = new_sd_ctx(params_.model_path.c_str(), "", "", "", "",
                                     params_.vae_path.c_str(), "", "", "", "", "",
                                     true, false, true, params_.n_threads,
                                     SD_TYPE_COUNT, CUDA_RNG, DEFAULT,
                                     false, false, false, false);

        if (!sd_ctx) {
            emit finished(false, "Failed to initialize SD context", "", args);
            return;
        }

        sd_image_t* results = nullptr;
        
        if (params_.mode == TXT2IMG) {
            results = txt2img(sd_ctx, params_.prompt.c_str(), params_.negative_prompt.c_str(),
                            -1, params_.cfg_scale, params_.guidance, 0.0f,
                            params_.width, params_.height, params_.sample_method,
                            params_.sample_steps, params_.seed, 1, nullptr, 0.9f,
                            20.0f, false, "", nullptr, 0, 0.0f, 0.01f, 0.2f);
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

            results = img2img(sd_ctx, input_image, mask_image, params_.prompt.c_str(),
                            params_.negative_prompt.c_str(), -1, params_.cfg_scale,
                            params_.guidance, 0.0f, params_.width, params_.height,
                            params_.sample_method, params_.sample_steps, params_.strength,
                            params_.seed, 1, nullptr, 0.9f, 20.0f, false, "",
                            nullptr, 0, 0.0f, 0.01f, 0.2f);
            free(input_buffer);
        }

        bool success = false;
        if (results && results[0].data) {
            success = stbi_write_png(params_.output_path.c_str(), results[0].width,
                                   results[0].height, results[0].channel,
                                   results[0].data, 0, nullptr);

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

            free(results[0].data);
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
        params.model_path = modelPath_->text().toStdString();
        params.vae_path = vaePath_->text().toStdString();
        params.output_path = outputPath_->text().toStdString();
        params.input_path = inputPath_->text().toStdString();
        params.prompt = prompt_->toPlainText().toStdString();
        params.negative_prompt = negativePrompt_->toPlainText().toStdString();
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

        jobQueue_.enqueue(params);
        updateJobsProgress();
        processNextJob();
    }

    void onGenerationFinished(bool success, const QString& message, const QString& imagePath, const QString& arguments) {
        completedJobs_++;
        updateJobsProgress();
        
        if (success) {
            addResultTab(imagePath, arguments);
            saveImageToSettings(imagePath, arguments);
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
            cancelBtn_->setEnabled(false);
            return;
        }
        
        progressBar_->setVisible(true);
        jobsProgressLabel_->setVisible(true);
        cancelBtn_->setEnabled(true);
        
        SDParams params = jobQueue_.dequeue();
        currentWorker_ = new GenerationWorker(params);
        connect(currentWorker_, &GenerationWorker::finished, this, &MainWindow::onGenerationFinished);
        currentWorker_->start();
    }
    
    void updateJobsProgress() {
        int total = completedJobs_ + jobQueue_.size() + (currentWorker_ ? 1 : 0);
        if (total == 0) total = 1;
        jobsProgressLabel_->setText(QString("Completed Jobs: %1/%2").arg(completedJobs_).arg(total));
        if (completedJobs_ == total) {
            completedJobs_ = 0;
        }
    }
    
    void cancelJob() {
        if (currentWorker_) {
            currentWorker_->terminate();
            currentWorker_->wait();
            currentWorker_->deleteLater();
            currentWorker_ = nullptr;
            progressBar_->setVisible(false);
            jobsProgressLabel_->setVisible(false);
            cancelBtn_->setEnabled(false);
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

        layout->addLayout(formLayout);

        // Generate button
        generateBtn_ = new QPushButton("Generate");
        generateBtn_->setMinimumHeight(40);
        layout->addWidget(generateBtn_);
        
        // Cancel button
        cancelBtn_ = new QPushButton("Cancel");
        cancelBtn_->setMinimumHeight(40);
        cancelBtn_->setEnabled(false);
        layout->addWidget(cancelBtn_);

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
        
        // Right side - results tabs
        tabWidget_ = new QTabWidget;
        tabWidget_->setTabsClosable(true);
        connect(tabWidget_, &QTabWidget::tabCloseRequested, this, &MainWindow::closeTab);
        mainLayout->addWidget(tabWidget_);
    }
    
    void addResultTab(const QString& imagePath, const QString& arguments) {
        auto* tabWidget = new QWidget;
        auto* tabLayout = new QVBoxLayout(tabWidget);
        
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
        
        // Arguments display
        auto* argsText = new QTextEdit;
        argsText->setPlainText(arguments);
        argsText->setMaximumHeight(150);
        argsText->setReadOnly(true);
        tabLayout->addWidget(argsText);
        
        QString tabName = QFileInfo(imagePath).fileName();
        int tabIndex = tabWidget_->addTab(tabWidget, tabName);
        tabWidget_->setTabToolTip(tabIndex, imagePath);
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
        connect(browseInputBtn_, &QPushButton::clicked, this, &MainWindow::browseInput);
        connect(browseOutputBtn_, &QPushButton::clicked, this, &MainWindow::browseOutput);
        connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onModeChanged);
        connect(generateBtn_, &QPushButton::clicked, this, &MainWindow::generate);
        connect(cancelBtn_, &QPushButton::clicked, this, &MainWindow::cancelJob);
    }

    void saveSettings() {
        QSettings settings;
        settings.setValue("mode", modeCombo_->currentIndex());
        settings.setValue("modelPath", modelPath_->text());
        settings.setValue("vaePath", vaePath_->text());
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
    }

    void loadSettings() {
        QSettings settings;
        qDebug() << "Loading from:" << settings.fileName();
        modeCombo_->setCurrentIndex(settings.value("mode", 0).toInt());
        modelPath_->setText(settings.value("modelPath").toString());
        vaePath_->setText(settings.value("vaePath").toString());
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
        onModeChanged();
        loadPersistedImages();
    }

    // UI elements
    QComboBox* modeCombo_;
    QLineEdit* modelPath_;
    QLineEdit* vaePath_;
    QLineEdit* inputPath_;
    QLineEdit* outputPath_;
    QPushButton* browseModelBtn_;
    QPushButton* browseVAEBtn_;
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
    QPushButton* generateBtn_;
    QPushButton* cancelBtn_;
    QProgressBar* progressBar_;
    QTabWidget* tabWidget_;
    QLabel* jobsProgressLabel_;
    
    QQueue<SDParams> jobQueue_;
    int completedJobs_ = 0;
    GenerationWorker* currentWorker_ = nullptr;
    
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