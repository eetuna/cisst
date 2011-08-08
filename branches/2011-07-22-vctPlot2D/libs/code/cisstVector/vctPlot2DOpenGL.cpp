/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: vctPlot2DGLBase.cpp 1238 2010-02-27 03:16:01Z auneri1 $

  Author(s):  Anton Deguet
  Created on: 2010-05-05

  (C) Copyright 2010-2011 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstVector/vctPlot2DOpenGL.h>

#if (CISST_OS == CISST_WINDOWS)
  #include <windows.h>
#endif

#if (CISST_OS == CISST_DARWIN)
  #include <OpenGL/gl.h>
#else
  #include <GL/gl.h>
#endif


vctPlot2DOpenGL::vctPlot2DOpenGL(void):
    vctPlot2DBase()
{
}


void vctPlot2DOpenGL::RenderInitialize(void)
{
    glMatrixMode(GL_MODELVIEW); // set the model view matrix
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST); // disable depth test
    glDisable(GL_LIGHTING); // disable lighting
    glShadeModel(GL_SMOOTH); // smooth render
    glClearColor(static_cast<GLclampf>(this->BackgroundColor[0]),
                 static_cast<GLclampf>(this->BackgroundColor[1]),
                 static_cast<GLclampf>(this->BackgroundColor[2]),
                 static_cast<GLclampf>(1.0));
}


void vctPlot2DOpenGL::RenderResize(double width, double height)
{

    const size_t numberOfScales = this->Scales.size();
    size_t scaleIndex ;

    this->Viewport.Assign(width, height);

    for(scaleIndex = 0 ; scaleIndex < numberOfScales ; scaleIndex++){
        vctPlot2DBase::Scale *scale = this->Scales[scaleIndex];
        scale->Viewport.Assign(width, height);       
    }
    
    GLsizei w = static_cast<GLsizei>(width);
    GLsizei h = static_cast<GLsizei>(height);

    glViewport(0 , 0, w ,h); // set up viewport
    glMatrixMode(GL_PROJECTION); // set the projection matrix
    glLoadIdentity();
    glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
}


void vctPlot2DOpenGL::Render(void)
{
    size_t scaleIndex;
    // Trace * trace;
    // Scale * scale;
    // size_t numberOfTraces;
    const size_t numberOfScales = this->Scales.size();
    // size_t numberOfPoints;

    // see if translation and scale need to be updated
    this->ContinuousUpdate();

    // clear
    glClear(GL_COLOR_BUFFER_BIT);

    // ------------------------------------------------------------------------------------------
    // those codes should be called for each Scale

    // make sure there is no left over transformation

    for(scaleIndex =0; 
        scaleIndex < numberOfScales; 
        scaleIndex++){   
        this->Render(Scales[scaleIndex]);
    }
}


//void vctPlot2DOpenGL::Render(const vctPlot2DBase::VerticalLine & line)
void vctPlot2DOpenGL::Render(const vctPlot2DBase::VerticalLine * line)
{
    // render vertical lines
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(this->Translation.X(), 0.0, 0.0);
    glScaled(this->ScaleValue.X(), 1.0, 1.0);

    // todo, should check for "visible" flag
    glBegin(GL_LINE_STRIP);
    glVertex2d(line->X, this->Viewport.Y());
    glVertex2d(line->X, 0);
    glEnd();
}

void vctPlot2DOpenGL::Render(const vctPlot2DBase::Trace * trace){

    size_t numberOfPoints;
    //trace->CriticalSectionForBuffer.Enter();
    if (trace->Visible) {
        numberOfPoints = trace->Data.size();
        glColor3d(trace->Color.Element(0),
                  trace->Color.Element(1),
                  trace->Color.Element(2));
        glLineWidth(static_cast<GLfloat>(trace->LineWidth));
        const double *data = trace->Data.Element(0).Pointer();
        size_t size = trace->Data.size();
        if (trace->IndexFirst >= trace->IndexLast) {
            // circular buffer is full/split in two
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(2, GL_DOUBLE, 0, data);
            // draw first part
            glDrawArrays(GL_LINE_STRIP,
                         static_cast<GLint>(trace->IndexFirst),
                         static_cast<GLsizei>(size - trace->IndexFirst));
            // draw second part
            glDrawArrays(GL_LINE_STRIP,
                         0,
                         static_cast<GLsizei>(trace->IndexLast + 1));
            glDisableClientState(GL_VERTEX_ARRAY);
            // draw between end of buffer and beginning
            glBegin(GL_LINE_STRIP);
            glVertex2d(trace->Data.Element(size - 1).X(),
                       trace->Data.Element(size - 1).Y());
            glVertex2d(trace->Data.Element(0).X(),
                       trace->Data.Element(0).Y());
            glEnd();
        } else {
            // simpler case, all points contiguous
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(2, GL_DOUBLE, 0, data);
            glDrawArrays(GL_LINE_STRIP,
                         0,
                         static_cast<GLsizei>(trace->IndexLast + 1));
            glDisableClientState(GL_VERTEX_ARRAY);
        }
    }

}


void vctPlot2DOpenGL::Render(const vctPlot2DBase::Scale * scale)
{

    const size_t numberOfTraces = scale->Traces.size();
    const size_t numberOfLines = scale->VerticalLines.size();
    size_t traceIndex, vlineIndex ;
    vctPlot2DBase::Trace *trace;
    vctPlot2DBase::VerticalLine *vline;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // glTranslated(this->Translation.X(), this->Translation.Y(), 0.0);
    // glScaled(this->ScaleValue.X(), this->ScaleValue.Y(), 1.0);    
    
    glTranslated(scale->Translation.X(), scale->Translation.Y(), 0.0);
    glScaled(scale->ScaleValue.X(), scale->ScaleValue.Y(), 1.0);
    
    for (traceIndex = 0;
         traceIndex < numberOfTraces;
         traceIndex++) {
        trace = scale->Traces[traceIndex];
        Render(trace);      
    }

    for(vlineIndex = 0; vlineIndex < numberOfLines; vlineIndex ++){
        vline = scale->VerticalLines[vlineIndex];
        Render(vline);
    }

}
