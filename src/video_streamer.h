#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

// Video streaming class
class VideoStreamer {
   private:
    int socket_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;
    bool is_running;
    std::thread streaming_thread;
    std::atomic<bool> new_frame_available;
    cv::Mat current_frame;
    std::mutex frame_mutex;
    int stream_port;
    std::string stream_type;

   public:
    VideoStreamer(int port = 8080, const std::string& type = "udp")
        : stream_port(port), stream_type(type), is_running(false), new_frame_available(false) {
        client_len = sizeof(client_addr);
    }

    ~VideoStreamer() {
        stop();
    }

    bool start() {
        if (stream_type == "udp") {
            return startUDPStreaming();
        } else if (stream_type == "http") {
            return startHTTPStreaming();
        }
        return false;
    }

    bool startUDPStreaming() {
        socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            std::cerr << "Error creating UDP socket" << std::endl;
            return false;
        }

        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(stream_port);

        if (bind(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Error binding UDP socket to port " << stream_port << std::endl;
            close(socket_fd);
            return false;
        }

        is_running = true;
        streaming_thread = std::thread(&VideoStreamer::udpStreamingLoop, this);
        std::cout << "UDP streaming started on port " << stream_port << std::endl;
        return true;
    }

    bool startHTTPStreaming() {
        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd < 0) {
            std::cerr << "Error creating HTTP socket" << std::endl;
            return false;
        }

        int opt = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(stream_port);

        if (bind(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Error binding HTTP socket to port " << stream_port << std::endl;
            close(socket_fd);
            return false;
        }

        if (listen(socket_fd, 5) < 0) {
            std::cerr << "Error listening on HTTP socket" << std::endl;
            close(socket_fd);
            return false;
        }

        is_running = true;
        streaming_thread = std::thread(&VideoStreamer::httpStreamingLoop, this);
        std::cout << "HTTP streaming started on port " << stream_port << std::endl;
        std::cout << "Open http://localhost:" << stream_port << "/stream in your browser" << std::endl;
        return true;
    }

    void updateFrame(const cv::Mat& frame) {
        if (frame.empty())
            return;

        std::lock_guard<std::mutex> lock(frame_mutex);

        // Ensure proper memory alignment and continuous memory layout
        if (frame.isContinuous()) {
            current_frame = frame.clone();
        } else {
            // Create a continuous copy if the frame is not continuous
            cv::Mat temp_frame;
            frame.copyTo(temp_frame);
            current_frame = temp_frame;
        }

        new_frame_available = true;
    }

    void stop() {
        is_running = false;
        if (streaming_thread.joinable()) {
            streaming_thread.join();
        }
        if (socket_fd >= 0) {
            close(socket_fd);
        }
    }

   private:
    void udpStreamingLoop() {
        std::vector<uchar> buffer;
        std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 60};  // Lower quality for stability

        // Wait for initial client connection with timeout
        char dummy_buffer[1];
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        std::cout << "Waiting for UDP client connection..." << std::endl;
        if (recvfrom(socket_fd, dummy_buffer, 1, 0, (struct sockaddr*)&client_addr, &client_len) < 0) {
            std::cout << "No UDP client connected, continuing without streaming..." << std::endl;
            return;
        }
        std::cout << "UDP client connected from " << inet_ntoa(client_addr.sin_addr) << std::endl;

        while (is_running) {
            if (new_frame_available.load()) {
                cv::Mat frame_to_send;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (!current_frame.empty() && current_frame.isContinuous()) {
                        frame_to_send = current_frame.clone();
                    }
                    new_frame_available = false;
                }

                if (!frame_to_send.empty()) {
                    // Resize frame for network efficiency
                    cv::Mat resized_frame;
                    if (frame_to_send.cols > 320) {
                        cv::resize(frame_to_send, resized_frame, cv::Size(320, 200));
                    } else {
                        resized_frame = frame_to_send;
                    }

                    // Encode frame as JPEG with error checking
                    buffer.clear();
                    try {
                        if (cv::imencode(".jpg", resized_frame, buffer, encode_params) && !buffer.empty()) {
                            // Limit max frame size
                            if (buffer.size() < 65536) {  // 64KB limit
                                // Send frame size first
                                uint32_t frame_size = htonl(buffer.size());
                                if (sendto(socket_fd, &frame_size, sizeof(frame_size), 0,
                                           (struct sockaddr*)&client_addr, client_len) < 0) {
                                    break;  // Client disconnected
                                }

                                // Send frame data in smaller chunks
                                const size_t chunk_size = 512;  // Smaller chunks for stability
                                size_t bytes_sent = 0;
                                while (bytes_sent < buffer.size() && is_running) {
                                    size_t remaining = buffer.size() - bytes_sent;
                                    size_t to_send = std::min(chunk_size, remaining);

                                    if (sendto(socket_fd, buffer.data() + bytes_sent, to_send, 0,
                                               (struct sockaddr*)&client_addr, client_len) < 0) {
                                        goto udp_loop_end;  // Break out of nested loops
                                    }
                                    bytes_sent += to_send;
                                }
                            }
                        }
                    } catch (const cv::Exception& e) {
                        std::cerr << "OpenCV encoding error: " << e.what() << std::endl;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // ~20 FPS for stability
        }

    udp_loop_end:
        std::cout << "UDP streaming ended" << std::endl;
    }

    void httpStreamingLoop() {
        while (is_running) {
            struct timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

            int client_socket = accept(socket_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;  // Timeout, try again
                }
                break;  // Real error
            }

            std::cout << "HTTP client connected from " << inet_ntoa(client_addr.sin_addr) << std::endl;

            // Send HTTP headers for MJPEG stream
            std::string headers =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Connection: keep-alive\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n\r\n";

            if (send(client_socket, headers.c_str(), headers.length(), MSG_NOSIGNAL) < 0) {
                close(client_socket);
                continue;
            }

            std::vector<uchar> buffer;
            std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 60};  // Lower quality for stability

            while (is_running) {
                if (new_frame_available.load()) {
                    cv::Mat frame_to_send;
                    {
                        std::lock_guard<std::mutex> lock(frame_mutex);
                        if (!current_frame.empty() && current_frame.isContinuous()) {
                            frame_to_send = current_frame.clone();
                        }
                        new_frame_available = false;
                    }

                    if (!frame_to_send.empty()) {
                        // Resize frame for network efficiency
                        cv::Mat resized_frame;
                        if (frame_to_send.cols > 320) {
                            cv::resize(frame_to_send, resized_frame, cv::Size(320, 200));
                        } else {
                            resized_frame = frame_to_send;
                        }

                        buffer.clear();
                        try {
                            if (cv::imencode(".jpg", resized_frame, buffer, encode_params) && !buffer.empty()) {
                                // Limit max frame size
                                if (buffer.size() < 65536) {  // 64KB limit
                                    std::string frame_header =
                                        "--frame\r\n"
                                        "Content-Type: image/jpeg\r\n"
                                        "Content-Length: " +
                                        std::to_string(buffer.size()) + "\r\n\r\n";

                                    if (send(client_socket, frame_header.c_str(), frame_header.length(), MSG_NOSIGNAL) < 0 ||
                                        send(client_socket, buffer.data(), buffer.size(), MSG_NOSIGNAL) < 0 ||
                                        send(client_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) {
                                        break;  // Client disconnected
                                    }
                                }
                            }
                        } catch (const cv::Exception& e) {
                            std::cerr << "OpenCV encoding error: " << e.what() << std::endl;
                            break;
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));  // ~20 FPS for stability
            }

            close(client_socket);
            std::cout << "HTTP client disconnected" << std::endl;
        }
    }
};
