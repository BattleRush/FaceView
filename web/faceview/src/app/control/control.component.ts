import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { ImageCroppedEvent } from 'ngx-image-cropper';


@Component({
    selector: 'app-control',
    templateUrl: './control.component.html',
    styleUrls: ['./control.component.scss']
})
export class ControlComponent implements OnInit {

    constructor(private httpClient: HttpClient) { }

    //currentVideoMode = VideoMode.File;
    //currentDetectionMode = DetectionMode.Nothing;

    // hardcoded
    livePath = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    //filePath = "/home/karlo/vid/vid.mp4"
    filePath = "I:\\Maturaarbeit\\vid.mp4";
    currentPath = "";
    baseUrl: string;
    lastImageLink: string;
    lastImageBase64: any;
    croppedBase64: string;

    httpConfig = { headers: new HttpHeaders().set('Content-Type', 'application/json') };

    currentSettings: Settings;
    //coordinates: any;
    coordinates = { x: 0 } // object that saves the crop coordinates

    loadedImage: boolean;

    ngOnInit() {
        // This avoids proxy
        this.baseUrl = "http://" + window.location.hostname + ":5000";
        //this.baseUrl = location.origin;

        this.lastImageLink = this.baseUrl + "/api/last_image";

        this.loadCurrentSettings();
    }


    public async loadCurrentSettings() {
        var response = this.httpClient
            .get(this.baseUrl + "/api/settings", this.httpConfig).subscribe(
                response => {
                    this.currentSettings = response as Settings;
                    console.log(this.currentSettings);
                },
                err => console.log(err)
            );
    }

    public async saveSettings() {
        console.log('hi');
        if (this.currentSettings.currentVideoMode == VideoMode.File || this.currentSettings.currentVideoMode == VideoMode.Image)
            this.currentPath = this.filePath;
        else if (this.currentSettings.currentVideoMode == VideoMode.Live)
            this.currentPath = this.livePath;



        var response = this.httpClient
            .post(this.baseUrl + "/api/settings", this.currentSettings, this.httpConfig).subscribe(
                response => console.log(response),
                err => console.log(err)
            );

    }

    public async startVid() {
        var response = this.httpClient
            .get(this.baseUrl + "/api/start", this.httpConfig).subscribe(
                response => console.log(response),
                err => console.log(err)
            );
    }



    public imageLoaded(event: any) {
        console.log(this.coordinates); // object that saves the crop coordinates

    }
    public imageCropped(event: ImageCroppedEvent) {
        this.croppedBase64 = event.base64;

        this.currentSettings.croppSettings.x = event.imagePosition.x1;
        this.currentSettings.croppSettings.y = event.imagePosition.y1;
        this.currentSettings.croppSettings.height = event.height;
        this.currentSettings.croppSettings.width = event.width;

        //this.currentSettings.croppSettings.origWidth = event.cropperPosition.x2 - event.cropperPosition.x1 + event.width;
        //this.currentSettings.croppSettings.origHeight = event.cropperPosition.y2 - event.cropperPosition.y1 + event.height;

        console.log("x: " + event.imagePosition.x1 + " | y: " + event.imagePosition.y1 + " | height: " + event.height + " | width: " + event.width);
        console.log(event);
        console.log(this.currentSettings);
    }

    // TODO Duplicate
    public async reloadImg() {
        this.lastImageLink = this.baseUrl + "/api/last_image?time=" + (new Date()).getTime();

        const result: any = await fetch(this.lastImageLink);
        const blob = await result.blob();
        let reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onload = () => {
            console.log(reader.result);
            this.lastImageBase64 = reader.result;
            this.loadedImage = true;

            var i = new Image();
            i.onload = () => {
                this.currentSettings.croppSettings.originalHeight = i.height;
                this.currentSettings.croppSettings.originalWidth = i.width;
            };

            if (typeof reader.result === 'string') {
                i.src = reader.result;
            }
        }

    }
}

export enum VideoMode {
    File = 0,
    Live = 1,
    Image = 2
}

export enum DetectionMode {
    Nothing = 0,
    DetectFaces = 1,
    RecognizeFaces = 2,
    Record = 3
}

export enum LiveResolution {
    Live_3264_2464 = 0,
    Live_3264_1848 = 1,
    Live_1920_1080 = 2,
    Live_1280_720 = 3
}

export class Settings {
    public videoPath: string;

    public currentVideoMode: VideoMode;
    public currentDetectionMode: DetectionMode;
    public croppSettings: CroppSettings;

    public liveResolution: LiveResolution;
}

export class CroppSettings {
    public x: number;
    public y: number;
    public height: number;
    public width: number;

    public originalHeight: number;
    public originalWidth: number;
}