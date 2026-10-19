#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// Select your board in Arduino IDE: AI Thinker ESP32-CAM
const char* WIFI_NAME = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://192.168.1.100:8000/api/read"; // PC IPv4 address
const char* METER_ID = "meter-01";
const unsigned long CAPTURE_INTERVAL_MS = 30000;

// AI Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

bool startCamera() {
  camera_config_t c;
  c.ledc_channel=LEDC_CHANNEL_0; c.ledc_timer=LEDC_TIMER_0;
  c.pin_d0=Y2_GPIO_NUM; c.pin_d1=Y3_GPIO_NUM; c.pin_d2=Y4_GPIO_NUM; c.pin_d3=Y5_GPIO_NUM;
  c.pin_d4=Y6_GPIO_NUM; c.pin_d5=Y7_GPIO_NUM; c.pin_d6=Y8_GPIO_NUM; c.pin_d7=Y9_GPIO_NUM;
  c.pin_xclk=XCLK_GPIO_NUM; c.pin_pclk=PCLK_GPIO_NUM; c.pin_vsync=VSYNC_GPIO_NUM; c.pin_href=HREF_GPIO_NUM;
  c.pin_sccb_sda=SIOD_GPIO_NUM; c.pin_sccb_scl=SIOC_GPIO_NUM; c.pin_pwdn=PWDN_GPIO_NUM; c.pin_reset=RESET_GPIO_NUM;
  c.xclk_freq_hz=20000000; c.pixel_format=PIXFORMAT_JPEG;
  c.frame_size=psramFound()?FRAMESIZE_UXGA:FRAMESIZE_SVGA;
  c.jpeg_quality=10; c.fb_count=psramFound()?2:1; c.grab_mode=CAMERA_GRAB_LATEST;
  return esp_camera_init(&c)==ESP_OK;
}

void sendCapture() {
  if(WiFi.status()!=WL_CONNECTED) return;
  camera_fb_t* frame=esp_camera_fb_get();
  if(!frame){ Serial.println("Capture failed"); return; }
  HTTPClient http; http.begin(SERVER_URL); http.setTimeout(20000);
  http.addHeader("Content-Type","image/jpeg"); http.addHeader("X-Meter-ID",METER_ID);
  int code=http.POST(frame->buf,frame->len);
  Serial.printf("Upload status: %d\n",code);
  if(code>0) Serial.println(http.getString());
  http.end(); esp_camera_fb_return(frame);
}

void setup(){
  Serial.begin(115200);
  if(!startCamera()){Serial.println("Camera initialization failed");return;}
  WiFi.begin(WIFI_NAME,WIFI_PASSWORD);
  while(WiFi.status()!=WL_CONNECTED){delay(500);Serial.print(".");}
  Serial.printf("\nConnected. ESP32 IP: %s\n",WiFi.localIP().toString().c_str());
  sendCapture();
}
void loop(){sendCapture();delay(CAPTURE_INTERVAL_MS);}
