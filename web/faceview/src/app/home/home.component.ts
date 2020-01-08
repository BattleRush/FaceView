import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { Chart, ChartColor } from 'chart.js'

// TODO move settings to a seperate file
import { Settings, DetectionMode, VideoMode, LiveResolution } from '../control/control.component';


//https://github.com/angular/angular/issues/20713
interface BeforeOnDestroy {
    ngxBeforeOnDestroy();
}

type NgxInstance = BeforeOnDestroy & Object;
type Descriptor = TypedPropertyDescriptor<Function>;
type Key = string | symbol;

function BeforeOnDestroy(target: NgxInstance, key: Key, descriptor: Descriptor) {
    return {
        value: async function (...args: any[]) {
            await target.ngxBeforeOnDestroy();
            return descriptor.value.apply(target, args);
        }
    }
}

@Component({
    selector: 'app-home',
    templateUrl: './home.component.html',
    styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

    constructor(private httpClient: HttpClient) {

    }

    httpConfig = { headers: new HttpHeaders().set('Content-Type', 'application/json') };

    chart: any;
    onlineChart: any;

    chartLoaded: boolean = false;

    // TODO Handle base url trough route
    baseUrl: string;
    videoFeedUrl: string;
    videoFeedFaceimageUrl: string;
    videoFeedFaceimageRecUrl: string;
    currentSettings: Settings;
    infoSubscription: Subscription;
    detectionsSubscription: Subscription;

    isDestroying: boolean = false;

    recognizedFaces: RecognizeFaceEvent[] = [];



    ngOnInit() {
        // This avoids proxy
        this.baseUrl = "http://" + window.location.hostname + ":5000";
        console.log(this.baseUrl);

        if (!this.chartLoaded) {
            this.createChart();//this.createChart();
            this.chartLoaded = true;
        }

        //this.baseUrl = location.origin;

        // 480 secs after 500 sec each thread is recycled
        this.detectionsSubscription = interval(2000).subscribe(i => this.loadDetectionsSubscription());
        this.infoSubscription = interval(480000).subscribe(i => this.loadInfos());
        this.loadInfos();
        //this.loadDetectionsSubscription();
        console.log(this.infoSubscription);



    }

    private async createChart() {
        this.chart = new Chart('canvas', {
            type: 'line',
            data: {
                //labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
                datasets: [/*{
                    label: 'My First dataset',
                    backgroundColor: "#F00",
                    borderColor: "#F00",
                    fill: false,
                    data: [
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor()
                    ],
                }*//*, {
                    label: 'My Second dataset',
                    backgroundColor: "#00F",
                    borderColor: "#00F",
                    fill: false,
                    data: [
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor(),
                        this.randomScalingFactor()
                    ],
                }*/
                ]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Online'
                },
                scales: {
                    xAxes: [{
                        display: true,
                        type: 'time',
                        distribution: 'linear',
                        time: {
                          //parser: 'MM/DD/YYYY HH:mm',
                          //tooltipFormat: 'll HH:mm',
                          unit: 'minute',
                          //unitStepSize: 1,
                          unitStepSize: 5,
                          stepSize: 5,
                          displayFormats: {
                            'millisecond': 'h:mm a',
                            'second': 'h:mm a',
                            'minute': 'h:mm a',
                            'hour': 'h:mm a',
                            'day': 'h:mm a',
                            'week': 'h:mm a',
                            'month': 'h:mm a',
                            'quarter': 'h:mm a',
                            'year': 'h:mm a'
                          }
                        }
                    }],
                    yAxes: [{
                        display: true,
                        //type: 'logarithmic',
                    }]
                }
            }
        });
    }

    private randomScalingFactor() {
        return Math.ceil(Math.random() * 10.0) * Math.pow(10, Math.ceil(Math.random() * 5));
    };

    public ngxBeforeOnDestroy() {
        console.log('1. BEFORE ONDESTROY INVOKE METHOD (await 2 sec)');

        this.isDestroying = true;
        return new Promise((resolve) => {
            setTimeout(() => this.endStream(), 2000);
        });

    }


    ngOnDestroy() {
        this.isDestroying = true;
        this.infoSubscription.unsubscribe();
        this.detectionsSubscription.unsubscribe();
        console.log("Destroying");
    }

    public loadDetectionsSubscription(){    
        this.loadDetectionsInfo();
    }


    public loadInfos() {
        this.loadCurrentSettings();

        if (!this.isDestroying) {
            this.videoFeedUrl = this.baseUrl + "/api/video_feed?time=" + (new Date()).getTime();
            this.videoFeedFaceimageUrl = this.baseUrl + "/api/video_feed_faceimage?time=" + (new Date()).getTime();
            this.videoFeedFaceimageRecUrl = this.baseUrl + "/api/video_feed_faceimage_rec?time=" + (new Date()).getTime();
        }
        else {

            this.videoFeedUrl = this.baseUrl + "/api/no_img";
            this.videoFeedFaceimageUrl = this.baseUrl + "/api/no_img";
            this.videoFeedFaceimageRecUrl = this.baseUrl + "/api/no_img";
        }
    }

    public endStream() {
        //this.infoSubscription.unsubscribe();


        // END all connections to backend

        console.log("ENDED ALL STREAMS");
    }

    // TODO Duplicate code
    public async loadCurrentSettings() {
        var response = this.httpClient.get(this.baseUrl + "/api/settings", this.httpConfig).subscribe(
                response => {
                    this.currentSettings = response as Settings;
                    console.log(this.currentSettings);
                },
                err => console.log(err)
            );
    }

    private getRandomColor() {
        var letters = '0123456789ABCDEF'.split('');
        var color = '#';
        for (var i = 0; i < 6; i++ ) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    public async loadDetectionsInfo() {
        if(!this.chartLoaded){
            return;
        }

        var response = this.httpClient
            .get(this.baseUrl + "/api/detections", this.httpConfig).subscribe(
                response => {
                    this.recognizedFaces = response as RecognizeFaceEvent[];


                    console.log(this.recognizedFaces);

                    //console.log(this.currentSettings);

                    var timeNow = new Date();

                    var minutes = 45;


                    //var labels = ["Now"];
                    var labels = [timeNow];

                    for (var i = 0; i < minutes; i++) {
                        timeNow = new Date(timeNow.getTime() - (1000*60)); // one min less
                     
                        labels.push(timeNow);
                    }

                    //this.chart.data.labels = labels;

                    for (var i = 0; i < this.recognizedFaces.length; i++) {
                        let faceFoundIndex = -1;

                        for (var d = 0; d < this.chart.data.datasets.length; d++) {

                            if (this.chart.data.datasets[d].label == this.recognizedFaces[i].name) {
                                faceFoundIndex = d;
                                break;
                            }

                        }




                        
                        var ret = this.recognizedFaces[i].groupedHits.sort((a,b) => a.time);
                        console.log(ret);


                        this.recognizedFaces[i].groupedHits = ret;

                        if(faceFoundIndex > -1){
                            // only update data
                            this.chart.data.datasets[d].data = [];
                            var result = [];

                            //console.log(this.recognizedFaces[i].groupedHits);

                            for (var h = 0; h < this.recognizedFaces[i].groupedHits.length; h++) {
                                var hit = this.recognizedFaces[i].groupedHits[h];
                                console.log(hit);
                                result.push({
                                    x: new Date(hit.time * 1000),
                                    y: hit.value
                                 });
                            }
                            this.chart.data.datasets[d].data = result;

                            //console.log(this.recognizedFaces[i].groupedHits);

                        }
                        else{
                            var result = [];

                            //console.log(this.recognizedFaces[i].groupedHits);

                            for (var h = 0; h < this.recognizedFaces[i].groupedHits.length; h++) {
                                var hit = this.recognizedFaces[i].groupedHits[h];
                                console.log(hit);
                                result.push({
                                    x: new Date(hit.time * 1000),
                                    y: hit.value
                                 });
                            }

                            var newDataset = {
                                label: this.recognizedFaces[i].name,
                                data: result,
                                borderColor : this.getRandomColor(),
                                fill: false,
                            };
                            this.chart.data.datasets.push(newDataset);
                            
                            //console.log("update");
                        }
                    }

                    this.chart.update();



                },
                err => console.log(err)
            );
    }
    /*export enum VideoMode {
        File = 0,
        Live = 1,
        Image = 2
    }
    
    export enum DetectionMode {
        Nothing = 0,
        DetectFaces = 1,
        RecognizeFaces = 2,
        Record = 3
    }*/

    public getVideoModeText(mode: VideoMode) {
        let modeText = "Unknown";
        switch (mode) {
            case VideoMode.File:
                modeText = "File";
                break;
            case VideoMode.Live:
                modeText = "Live";
                break;
            case VideoMode.Image:
                modeText = "Image";
                break;
        }
        return modeText;
    }

    public getDetectionModeText(mode: DetectionMode) {
        let modeText = "Unknown";
        switch (mode) {
            case DetectionMode.Nothing:
                modeText = "Nothing";
                break;
            case DetectionMode.DetectFaces:
                modeText = "DetectFaces";
                break;
            case DetectionMode.RecognizeFaces:
                modeText = "RecognizeFaces";
                break;
            case DetectionMode.Record:
                modeText = "Record";
                break;
        }
        return modeText;
    }


    // TODO: Duplicate
    public getLiveResolutionText(mode: LiveResolution) {
        let modeText = "Unknown";
        switch (mode) {
            case LiveResolution.Live_3264_2464:
                modeText = "3264 x 2464 (102 : 77) @5 fps";
                break;
            case LiveResolution.Live_3264_1848:
                modeText = "3264 x 1848 (136 : 77) @5 fps";
                break;
            case LiveResolution.Live_1920_1080:
                modeText = "1920 x 1080 (16 : 9) @15 fps";
                break;
            case LiveResolution.Live_1280_720:
                modeText = "1280 x 720  (16 : 9) @30 fps";
                break;
        }
        return modeText;
    }
}

/*
interface KeyValuePair {
    key: number;
    value: number;
}*/

export class GroupInfo {
    time: number;
    value: number;
}

export class RecognizeFaceEvent {
    public faceId: string;
    public name: string;
    public isOnline: boolean;
    public groupedHits: GroupInfo[];

    public allRecognizedAt: number[];
    public allProbabilities: number[];
}