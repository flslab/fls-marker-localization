#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <lgpio.h>
#include <string>

// On Raspberry Pi 5 / CM5, the primary 40-pin header is managed by the RP1 chip, 
// which is typically exposed as gpiochip4.
const int GPIO_CHIP = 4; 

// The single control pin connected to the MOSFET gate driving all 4 LEDs
const int CONTROL_PIN = 17;

// Global flag for clean exit on Ctrl+C
volatile sig_atomic_t keep_running = 1;

void sig_handler(int sig) {
    keep_running = 0;
}

// Generate the OOK packet array: [1 Start] + [0 Sync] + [Payload] + [5 Rest]
std::vector<int> build_packet(uint16_t marker_id, int payload_size) {
    std::vector<int> packet;
    
    // 1. Start Bit (High)
    packet.push_back(1);
    
    // 2. Sync Bit (Low) - Creates a falling edge
    packet.push_back(0);
    
    // 3. Payload: ID (MSB first)
    for (int i = payload_size - 1; i >= 0; --i) {
        packet.push_back((marker_id >> i) & 1);
    }
    
    // 4. Rest Period (High) - 4 bit-times to separate transmissions and increase tracking frames
    for (int i = 0; i < 4; ++i) {
        packet.push_back(1);
    }
    
    return packet;
}

int main(int argc, char* argv[]) {
    double fps = 50.0; // default
    int my_marker_id = 0;
    int payload_size = 4;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fps" && i + 1 < argc) {
            fps = std::stod(argv[++i]);
        } else if (arg == "--marker-id" && i + 1 < argc) {
            my_marker_id = std::stoi(argv[++i]);
        } else if (arg == "--payload-size" && i + 1 < argc) {
            payload_size = std::stoi(argv[++i]);
        }
    }
    // 1/fps * 3 seconds = 3000/fps milliseconds
    auto bit_duration = std::chrono::milliseconds(static_cast<int>(3000.0 / fps));

    // Register signal handler for clean shutdown
    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    // Open the GPIO chip
    int handle = lgGpiochipOpen(GPIO_CHIP);
    if (handle < 0) {
        std::cerr << "Failed to open gpiochip" << GPIO_CHIP << " (Error " << handle << ")\n";
        return 1;
    }

    // Claim the single control pin as output
    if (lgGpioClaimOutput(handle, 0, CONTROL_PIN, 0) < 0) {
        std::cerr << "Failed to claim GPIO " << CONTROL_PIN << "\n";
        lgGpiochipClose(handle);
        return 1;
    }

    if (my_marker_id == 0) {
        std::cout << "Marker ID is 0. LEDs will remain ON without blinking.\n";
        std::cout << "Press Ctrl+C to stop.\n";
        
        lgGpioWrite(handle, CONTROL_PIN, 1);
        while (keep_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else {
        std::vector<int> packet = build_packet(static_cast<uint16_t>(my_marker_id), payload_size);

        std::cout << "Broadcasting Marker ID " << my_marker_id << " (Payload: " << payload_size << " bits) at " << fps << " Hz...\n";
        std::cout << "Press Ctrl+C to stop.\n";

        // Setup absolute timing baseline
        auto next_tick = std::chrono::steady_clock::now();

        while (keep_running) {
            for (int bit : packet) {
                if (!keep_running) break;

                // Write the current bit to the single control pin
                lgGpioWrite(handle, CONTROL_PIN, bit);

                // Calculate the exact time the NEXT bit should occur
                next_tick += bit_duration;

                // Sleep exactly until that absolute time to prevent drift
                std::this_thread::sleep_until(next_tick);
            }
        }
    }

    // Clean up: turn off the pin and release it
    std::cout << "\nShutting down...\n";
    lgGpioWrite(handle, CONTROL_PIN, 0);
    lgGpiochipClose(handle);

    return 0;
}