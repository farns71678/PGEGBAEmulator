//#define M_CORE_GBA

#include "olcPGEX_MiniAudio.h"
#include "olcPixelGameEngine.h"
#include "mgba-plugin/mgba_plugin.h"
#include "miniaudio.h"
#include <vector>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <chrono>
/**
 * source "/workspaces/PGEGBAEmulator/lib/emsdk/emsdk_env.sh"
 * 
 * To Build:
 * emcmake cmake . -B emscripten-build
 * cmake --build emscripten-build --target GBAEmulatorApp -- -j 8
 */

const auto SCREEN_HEIGHT = 160;
const auto  SCREEN_WIDTH = 240;
constexpr auto SCREEN_RATIO = SCREEN_WIDTH / (float)SCREEN_HEIGHT;
const float frameTime = 1.0f/ 60.0f;

const uint16_t MAX_KEYS = 10;
const char* keys[] = {
    "A",
    "B",
    "Select",
    "Start",
    "Right",
    "Left",
    "Up",
    "Down",
    "R",
    "L"
};

std::map<std::string, olc::Key> keyMap = {
    {"A", olc::Z},
    {"B", olc::X},
    {"Select", olc::SHIFT},
    {"Start", olc::ENTER},
    {"Right", olc::RIGHT},
    {"Left", olc::LEFT},
    {"Up", olc::UP},
    {"Down", olc::DOWN},
    {"R", olc::E},
    {"L", olc::Q}
};

std::vector<uint8_t> readFile(const char* filepath);

class GBAEmulatorApp : public olc::PixelGameEngine
{
public:
    int instanceId = -1;
    size_t frameCount = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    olc::Sprite gbaScreen;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastStatsTime;

private:
    float accumulator = 0.0f;
    // audio ring buffer (single-producer, single-consumer)
    // Make the ring buffer larger to reduce underflow risk. This is the number of float samples (interleaved).
    // At GBA_SAMPLE_RATE=32768 and 2 channels, 1<<16 samples ~= 1 second of audio. Increase to 2 seconds.
    static constexpr size_t RING_BUFFER_SAMPLES = 1 << 17; // power of two (~2s)
    static constexpr size_t RING_BUFFER_MASK = RING_BUFFER_SAMPLES - 1;
    std::vector<float> audioRingBuffer;
    std::atomic<uint64_t> rbWriteIndex{0};
    std::atomic<uint64_t> rbReadIndex{0};
    ma_device audioDevice;
    bool audioInitialized = false;
    uint32_t audioChannels = GBA_NUM_CHANNELS;
    uint32_t audioSampleRate = GBA_SAMPLE_RATE;
    std::vector<float> tempAudioFetch;
    std::atomic<uint64_t> underflowCount{0};
    std::atomic<uint64_t> overrunCount{0};

    // Resampling state (produce device sample rate from emulator sample rate)
    double resamplePosFrames = 0.0;         // fractional source-frame position relative to rbReadIndex
    uint32_t producerSampleRate = GBA_SAMPLE_RATE; // source/emulator sample rate (Hz)
    uint32_t deviceSampleRate = 0;         // actual device sample rate (Hz), discovered at runtime

public:

    GBAEmulatorApp()
    {
        sAppName = "olc::PixelGameEngine GBA Emulator ";
        this->instanceId = 0;
        audioRingBuffer.resize(RING_BUFFER_SAMPLES);
        tempAudioFetch.reserve(8192);
        lastStatsTime = std::chrono::high_resolution_clock::now();
    }

    bool OnUserCreate() override
    {
        // Initialization code here
        startTime = std::chrono::high_resolution_clock::now();
        gbaScreen = olc::Sprite(SCREEN_WIDTH, SCREEN_HEIGHT);

        // load rom
        std::vector<uint8_t> romData = readFile("assets/roms/pokemon-emerald-version.gba");
        if (romData.empty()) {
            return false;
        }

        // load bios
        std::vector<uint8_t> biosData = readFile("assets/bios/gba_bios.bin");
        if (biosData.empty()) {
            return false;
        }

    // pass the BIOS buffer we loaded above (was previously nullptr)
    this->instanceId = createInstance(romData.data(), romData.size(), biosData.data(), biosData.size(), "assets/saves/pokeemerald.sav");
        if (this->instanceId == -1) {
            return false;
        }
        // Initialize miniaudio device with ring-buffer callback
        audioChannels = GBA_NUM_CHANNELS;
        // producer sample rate is whatever the emulator reports
        producerSampleRate = getSampleRate(this->instanceId);
        audioSampleRate = producerSampleRate; // keep legacy variable populated

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = audioChannels;
        // Prefer the system default device sample rate; set 0 to let miniaudio pick default if available.
        // If the device ends up using a different sample-rate than the producer (GBA), we'll resample in the callback.
        config.sampleRate = 0;
        config.dataCallback = [](ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
            (void)pInput;
            GBAEmulatorApp* app = (GBAEmulatorApp*)pDevice->pUserData;
            float* out = (float*)pOutput;
            const size_t outFrames = (size_t)frameCount;
            const size_t channels = (size_t)app->audioChannels;

            // Discover device sample rate once
            uint32_t devRate = 0;
            if (pDevice && pDevice->sampleRate > 0) {
                devRate = (uint32_t)pDevice->sampleRate;
            } else if (app->deviceSampleRate > 0) {
                devRate = app->deviceSampleRate;
            } else {
                // fallback to producer rate if unknown
                devRate = app->producerSampleRate;
            }
            if (app->deviceSampleRate == 0) app->deviceSampleRate = devRate;

            const double srcRate = (double)app->producerSampleRate;
            const double dstRate = (double)devRate;
            const double srcPerDst = srcRate / dstRate; // how many source frames advance per 1 output frame

            // snapshot indices
            uint64_t r = app->rbReadIndex.load(std::memory_order_acquire);
            uint64_t w = app->rbWriteIndex.load(std::memory_order_acquire);
            uint64_t availSamples = (w >= r) ? (w - r) : 0;
            uint64_t availFrames = availSamples / channels;

            // local position measured in source frames (relative to r)
            double pos = app->resamplePosFrames;

            size_t outFrame = 0;
            for (; outFrame < outFrames; ++outFrame) {
                // required source frame indices
                double srcPos = pos;
                uint64_t i0 = (uint64_t)floor(srcPos);
                uint64_t i1 = i0 + 1;

                if (i1 >= availFrames) {
                    // not enough source frames to interpolate pDevice further -> fill remaining with silence
                    size_t remainingSamples = (outFrames - outFrame) * channels;
                    memset(out + outFrame * channels, 0, remainingSamples * sizeof(float));
                    break;
                }

                double frac = srcPos - (double)i0;

                // read samples for frame i0 and i1, channel-by-channel
                for (size_t ch = 0; ch < channels; ++ch) {
                    uint64_t sampleIndex0 = r + (i0 * channels) + ch;
                    uint64_t sampleIndex1 = r + (i1 * channels) + ch;
                    size_t idx0 = (size_t)(sampleIndex0 & RING_BUFFER_MASK);
                    size_t idx1 = (size_t)(sampleIndex1 & RING_BUFFER_MASK);
                    float s0 = app->audioRingBuffer[idx0];
                    float s1 = app->audioRingBuffer[idx1];
                    out[outFrame * channels + ch] = (float)(s0 * (1.0 - frac) + s1 * frac);
                }

                pos += srcPerDst;
            }

            // Advance read index by the number of whole source frames consumed
            uint64_t consumedFrames = (uint64_t)floor(pos);
            if (consumedFrames > 0) {
                app->rbReadIndex.store(r + consumedFrames * channels, std::memory_order_release);
                pos -= (double)consumedFrames;
            }

            // store fractional remainder for next callback
            app->resamplePosFrames = pos;

            // If we broke out early due to insufficient source frames, we should count underflows
            if (outFrame < outFrames) {
                app->underflowCount.fetch_add((uint64_t)(outFrames - outFrame), std::memory_order_relaxed);
            }
        };
        config.pUserData = this;

        ma_result result = ma_device_init(nullptr, &config, &audioDevice);
        if (result != MA_SUCCESS) {
            fprintf(stderr, "Failed to init audio device: %d\n", result);
        } else {
            // Prefill ring buffer with ~100ms of audio to avoid immediate underflow when the device starts.
            const double prefillSeconds = 0.100; // 100 ms
            const uint64_t samplesNeeded = (uint64_t)(producerSampleRate * audioChannels * prefillSeconds);
            uint64_t startW = rbWriteIndex.load(std::memory_order_acquire);
            uint64_t startR = rbReadIndex.load(std::memory_order_acquire);
            // Run a few frames until we've produced enough audio or a max iteration is reached.
            int maxIters = 120; // don't spin forever
            while ((rbWriteIndex.load(std::memory_order_acquire) - rbReadIndex.load(std::memory_order_acquire)) < samplesNeeded && maxIters-- > 0) {
                updateSingleFrame(this->instanceId);
                uint32_t sampleCount = getNumSamplesAvailable(this->instanceId);
                if (sampleCount > 0) {
                    tempAudioFetch.resize(sampleCount);
                    uint32_t got = getCurrentFrameSoundSamples(this->instanceId, sampleCount, tempAudioFetch.data());
                    if (got > 0) {
                        uint64_t w = rbWriteIndex.load(std::memory_order_acquire);
                        uint64_t r = rbReadIndex.load(std::memory_order_acquire);
                        uint64_t free = RING_BUFFER_SAMPLES - (w - r);
                        if (got > free) {
                            // drop oldest samples to make room
                            rbReadIndex.store(r + (got - free), std::memory_order_release);
                            overrunCount.fetch_add(1, std::memory_order_relaxed);
                            r = rbReadIndex.load(std::memory_order_acquire);
                        }
                        size_t toWrite = got;
                        size_t first = std::min(toWrite, RING_BUFFER_SAMPLES - (size_t)(w & RING_BUFFER_MASK));
                        memcpy(&audioRingBuffer[(size_t)(w & RING_BUFFER_MASK)], tempAudioFetch.data(), first * sizeof(float));
                        if (first < toWrite) {
                            size_t second = toWrite - first;
                            memcpy(&audioRingBuffer[0], tempAudioFetch.data() + first, second * sizeof(float));
                        }
                        rbWriteIndex.store(w + toWrite, std::memory_order_release);
                    }
                }
            }

            // Now start the device
            result = ma_device_start(&audioDevice);
            if (result == MA_SUCCESS) {
                audioInitialized = true;
                // record actual device sample rate for resampler if available
                if (audioDevice.sampleRate > 0) {
                    deviceSampleRate = (uint32_t)audioDevice.sampleRate;
                }
            } else {
                fprintf(stderr, "Failed to start audio device: %d\n", result);
            }
        }
        
        return true;
    }

    bool OnUserUpdate(float fElapsedTime) override
    {
        // Main update loop code here
        accumulator += fElapsedTime;
        while (accumulator >= frameTime) {
            // Handle input
            uint16_t keyState = 0;
            for (int i = 0; i < MAX_KEYS; ++i) {
                if (GetKey(keyMap[keys[i]]).bHeld) {
                    keyState |= (1 << i);
                }
            }

            pushKeys(this->instanceId, keyState);
            /*frameCount++;
            double rate = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startTime).count();
            std::cout << ("Update frame: " + std::to_string(frameCount) + "\t Rate: " + std::to_string(frameCount / (rate + 1e-6)) + " fps\n");*/
            updateSingleFrame(this->instanceId);

            // get audio samples and push into ring buffer
            uint32_t sampleCount = getNumSamplesAvailable(this->instanceId);
            if (sampleCount > 0) {
                tempAudioFetch.resize(sampleCount);
                uint32_t got = getCurrentFrameSoundSamples(this->instanceId, sampleCount, tempAudioFetch.data());
                if (got > 0) {
                    uint64_t w = rbWriteIndex.load(std::memory_order_acquire);
                    uint64_t r = rbReadIndex.load(std::memory_order_acquire);
                    uint64_t free = RING_BUFFER_SAMPLES - (w - r);
                    if (got > free) {
                        // drop oldest samples to make room
                        rbReadIndex.store(r + (got - free), std::memory_order_release);
                        r = rbReadIndex.load(std::memory_order_acquire);
                    }
                    size_t toWrite = got;
                    size_t first = std::min(toWrite, RING_BUFFER_SAMPLES - (size_t)(w & RING_BUFFER_MASK));
                    memcpy(&audioRingBuffer[(size_t)(w & RING_BUFFER_MASK)], tempAudioFetch.data(), first * sizeof(float));
                    if (first < toWrite) {
                        size_t second = toWrite - first;
                        memcpy(&audioRingBuffer[0], tempAudioFetch.data() + first, second * sizeof(float));
                    }
                    rbWriteIndex.store(w + toWrite, std::memory_order_release);
                }
            }

                // Print occasional audio stats (once per second)
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastStatsTime).count() >= 1) {
                    uint64_t w = rbWriteIndex.load(std::memory_order_acquire);
                    uint64_t r = rbReadIndex.load(std::memory_order_acquire);
                    uint64_t filled = w - r;
                    fprintf(stderr, "audio: buffer filled=%zu samples underflow=%zu overrun=%zu\n", (size_t)filled, (size_t)underflowCount.load(), (size_t)overrunCount.load());
                    lastStatsTime = now;
                }

            // Render frame
            uint8_t* pixels = getCurrentFramePixels(this->instanceId);
            
            if (pixels) {
                for (int y = 0; y < SCREEN_HEIGHT; ++y) {
                    for (int x = 0; x < SCREEN_WIDTH; ++x) {
                        uint32_t color = ((uint32_t*)pixels)[y * SCREEN_WIDTH + x];

                        // pixel color is little endian (I think)
                        this->gbaScreen.SetPixel(x, y, olc::Pixel(color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF));
                        //Draw(x, y, olc::Pixel((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF));
                    }
                }
            }

            DrawSprite(0, 0, &this->gbaScreen);

            accumulator -= frameTime;
        }
        return true;
    }

    bool OnUserDestroy() override
    {
        // Cleanup code here
        releaseInstance(this->instanceId);
        if (audioInitialized) {
            ma_device_uninit(&audioDevice);
            audioInitialized = false;
        }
        std::cout << "Emulator instance released." << std::endl;
        return true;
    }

};

int main() {
    GBAEmulatorApp app;

    if (app.Construct(SCREEN_WIDTH, SCREEN_HEIGHT, 2, 2)) {
        app.Start();
    }
    

    return 0;
}

std::vector<uint8_t> readFile(const char* filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return {};
    }
    file.unsetf(std::ios::skipws);
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> vec;
    vec.reserve(fileSize);

    vec.insert(vec.begin(),
        std::istream_iterator<uint8_t>(file),
        std::istream_iterator<uint8_t>());

    return vec;
}