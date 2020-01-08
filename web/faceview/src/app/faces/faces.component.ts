import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ImageCroppedEvent } from 'ngx-image-cropper';


@Component({
    selector: 'app-faces',
    templateUrl: './faces.component.html',
    styleUrls: ['./faces.component.scss']
})
export class FacesComponent implements OnInit {

    constructor(private httpClient: HttpClient) { }

    @ViewChild('fileInput', { static: false })
    fileInputElement: ElementRef;


    baseUrl: string;

    // TODO global
    httpConfig = { headers: new HttpHeaders().set('Content-Type', 'application/json') };

    ngOnInit() {
        this.baseUrl = "http://" + window.location.hostname + ":5000";

        this.loadAllFaces();
    }
    imageChangedEvent: any = '';
    croppedImage: any;
    newFaceName: string;
    imageBase64String: any;

    fileChangeEvent(event: any): void {
        console.log(event);
        this.imageChangedEvent = event;
        this.imageReady = true;
    }
    imageCropped(event: ImageCroppedEvent) {
        this.croppedImage = event.base64;
    }
    
    allFaces: Face[];

    imageReady: boolean;

    public async loadAllFaces() {
        var response = this.httpClient
            .get(this.baseUrl + "/api/faces", this.httpConfig).subscribe(
                response => {
                    console.log(response);
                    this.allFaces = response as Face[];

                },
                err => console.log(err)
            );
    }

    public async saveFace(newFace: Face) {
        var response = this.httpClient
            .post(this.baseUrl + "/api/faces", newFace, this.httpConfig).subscribe(
                response => console.log(response),
                err => console.log(err)
            );

    }
    public async deleteFace(faceId: string) {
        
        var response = this.httpClient
            .delete(this.baseUrl + "/api/faces?faceId="+faceId, this.httpConfig).subscribe(
                response => console.log(response),
                err => console.log(err)
            );

    }

    public addNewFace() {
        console.log(this.allFaces);
        
        if(this.newFaceName == null || this.croppedImage == null) {
            return;
        }

        let newFace = new Face(this.newFaceName, this.croppedImage);

        // TODO check if save is successful
        this.saveFace(newFace);

        this.allFaces.push(newFace);
        this.imageReady = false;

        
        console.log(this.allFaces);
        this.newFaceName = null;
        this.croppedImage = null;

        console.log(this.fileInputElement);
        this.fileInputElement.nativeElement.value = "";
    }

    public removeFace(face){
        this.deleteFace(face.faceId);
        this.allFaces.splice(this.allFaces.indexOf(face), 1);
    }

        // TODO Duplicate
    public async loadLastImage() {
            var lastImageLink = this.baseUrl + "/api/last_image?time=" + (new Date()).getTime();
    
            const result: any = await fetch(lastImageLink);
            const blob = await result.blob();
            let reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onload = () => {
                console.log(reader.result);
                this.imageReady = true;
                this.imageBase64String = reader.result;
          
    
            }
    
        }
}

export class Face {    
    constructor(name: string, imageBase64: string) {
        this.name = name;
        this.imageBase64 = imageBase64;
    }
    public name: string;
    public imageBase64: string;
    public lastSeen: Date;
}