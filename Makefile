# Makefile for Stable Diffusion Qt6 Example

# Variables
BUILD_DIR = build
TARGET = sd-qt6
CMAKE_BUILD_TYPE ?= Release

# Qt6 detection
QT6_FOUND := $(shell pkg-config --exists Qt6Core Qt6Widgets && echo "yes" || echo "no")

.PHONY: all configure build clean install help

# Default target
all: build

# Configure the build system
configure:
	@echo "Configuring build system..."
ifeq ($(QT6_FOUND),no)
	@echo "Error: Qt6 not found. Please install Qt6 development packages."
	@echo "On macOS: brew install qt6"
	@echo "On Ubuntu/Debian: sudo apt install qt6-base-dev"
	@exit 1
endif
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DSD_BUILD_EXAMPLES=ON ..

# Build the project
build: configure
	@echo "Building $(TARGET)..."
	cd $(BUILD_DIR) && $(MAKE) $(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

# Install the built binary
install: build
	@echo "Installing $(TARGET)..."
	cd $(BUILD_DIR) && $(MAKE) install

run:
	@echo "Running $(TARGET)..."
	$(BUILD_DIR)/bin/$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  configure    - Configure the build system"
	@echo "  build        - Build the Qt6 application (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install the built application"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CMAKE_BUILD_TYPE - Build type (Release, Debug, RelWithDebInfo, MinSizeRel)"
	@echo "                     Default: Release"
	@echo ""
	@echo "Examples:"
	@echo "  make                           # Build with default settings"
	@echo "  make CMAKE_BUILD_TYPE=Debug    # Build debug version"
	@echo "  make clean                     # Clean build directory"