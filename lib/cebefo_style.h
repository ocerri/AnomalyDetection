#include <TROOT.h>
#include <TSystem.h>
#include <TStyle.h>
// #include <TCanvas.h>
// #include <TPad.h>
// #include <TH2.h>
// #include <TNamed.h>
// #include <TGraph.h>
// #include <TGaxis.h>
// #include <TF1.h>
// #include <TLatex.h>
// #include <TMath.h>
// #include <TLegend.h>
// #include <TPaveStats.h>

// Drawing style setup function
#ifndef _CEBEFO_STYLE
#define _CEBEFO_STYLE

void cebefo_style()
{
	//TStyle *gStyle = new TStyle("gStyle","gStyle");

	//our settings, maybe in the future we could create a more personal style
	gStyle->SetPadTickX(1);
	gStyle->SetPadTickY(1);
	gStyle->SetHistLineWidth(3);
	gStyle->SetMarkerStyle(1);

	//gStyle->SetTextSize(0.065);

	gStyle->SetOptFit(1111);
	// axis
	gStyle->SetTitleSize(.05,"X");//.055
	gStyle->SetTitleOffset(1.1,"X");//1.2,0.9
	//gStyle->SetLabelOffset(0.003,"X");
	gStyle->SetLabelSize(.05,"X");
	//gStyle->SetLabelFont(42,"X");

	gStyle->SetTitleSize(.05,"Y");//.055
	gStyle->SetTitleOffset(1.1,"Y");
	//gStyle->SetLabelOffset(0.008,"Y");
	gStyle->SetLabelSize(.05,"Y");
	//gStyle->SetLabelFont(42,"Y");

	gStyle->SetPadLeftMargin(.16);
	gStyle->SetPadBottomMargin(.12);

	gStyle->SetTitleSize(.05,"Z");
	gStyle->SetTitleOffset(1.8,"Z");
	//gStyle->SetLabelOffset(0.008,"Z");
	gStyle->SetLabelSize(0.06,"Z");
	//gStyle->SetLabelFont(42,"Z");

	//gStyle->SetTitleSize(.2,"abc");
	//gStyle->SetTitleFontSize(0.9);
	// gStyle->SetLegendTextSize(0.04);
	//gStyle->SetStatFontSize(0.2);

	//histograms properties
	gStyle->SetOptStat(112210);


	//gStyle.SetStripDecimals(False)
	//gStyle.SetLineStyleString(11,"20 10")


	// canvas
	//gStyle->SetCanvasColor(0);
	//gStyle->SetCanvasBorderSize(10);
	//gStyle->SetCanvasBorderMode(0);
	//gStyle->SetCanvasDefW(600);
	//gStyle->SetCanvasDefH(600);

	// pads
	// gStyle->SetPadColor(0);
	// gStyle->SetPadBorderSize(10);
	// gStyle->SetPadBorderMode(0);
	gStyle->SetPadLeftMargin(.12);
	gStyle->SetPadRightMargin(.02);
	gStyle->SetPadBottomMargin(.12);
	gStyle->SetPadTopMargin(.07);
	gStyle->SetPadGridX(1);
	gStyle->SetPadGridY(1);
	// gStyle->SetPadTickX(1);
	// gStyle->SetPadTickY(1);
	//
	// // frame
	// gStyle->SetFrameBorderMode(0);
	// gStyle->SetFrameBorderSize(10);
	// gStyle->SetFrameFillStyle(0);
	// gStyle->SetFrameFillColor(0);
	// gStyle->SetFrameLineColor(1);
	// gStyle->SetFrameLineStyle(0);
	// gStyle->SetFrameLineWidth(1);
	//
	// // histogram
	// gStyle->SetHistFillColor(0);
	// gStyle->SetHistFillStyle(1001);// solid
	// gStyle->SetHistLineColor(1);
	// gStyle->SetHistLineStyle(0);
	// gStyle->SetHistLineWidth(3);
	// gStyle->SetOptStat(0);
	// gStyle->SetOptFit(0);
	// gStyle->SetStatColor(0);
	// gStyle->SetStatBorderSize(1);
	// gStyle->SetStatFontSize(.05);
	//
	// // graph
	// gStyle->SetEndErrorSize(0);
    // gStyle->SetErrorX(0.5);
	//
	// // marker
	// gStyle->SetMarkerStyle(20);
	// gStyle->SetMarkerColor(kBlack);
	// gStyle->SetMarkerSize(0.6);

	// title
	// gStyle->SetOptTitle(1);
	// gStyle->SetTitleX(.15);
	// gStyle->SetTitleFillColor(0);
	// gStyle->SetTitleBorderSize(0);
	// gStyle->SetTitleStyle(0);

	// // axis
	// gStyle->SetNdivisions(505,"X");
	// gStyle->SetNdivisions(505,"Y");
	//
	// gStyle->SetTitleSize(.05,"X");//.055
	// gStyle->SetTitleOffset(1.,"X");//1.2,0.9
	// gStyle->SetLabelOffset(0.003,"X");
	// gStyle->SetLabelSize(.05,"X");
	// gStyle->SetLabelFont(42,"X");
	//
	// gStyle->SetTitleSize(.05,"Y");//.055
	// gStyle->SetTitleOffset(1.8,"Y");
	// gStyle->SetLabelOffset(0.008,"Y");
	// gStyle->SetLabelSize(.05,"Y");
	// gStyle->SetLabelFont(42,"Y");
	//
	// //gStyle->SetStripDecimals(kFALSE);
	//
	// gStyle->SetTitleSize(0.05,"Z");
	// gStyle->SetTitleOffset(1.800,"Z");
	// gStyle->SetLabelOffset(0.008,"Z");
	// gStyle->SetLabelSize(0.05,"Z");
	// gStyle->SetLabelFont(42,"Z");
	//
	// // fonts
	// gStyle->SetTextSize(.05);//.055
	// gStyle->SetTextFont(42);
	// gStyle->SetStatFont(42);
	// gStyle->SetTitleFont(42,"");
	// gStyle->SetTitleFont(42,"Z");
	// gStyle->SetTitleFont(42,"X");
	// gStyle->SetTitleFont(42,"Y");
	//
	// // function
	// gStyle->SetFuncColor(kBlue);
	// gStyle->SetFuncStyle(0);
	// gStyle->SetFuncWidth(1);
	//
	// // palette
	// gStyle->SetPalette(1);
	// gStyle->SetNumberContours(20);

	// set gStyle as current style
	//gROOT->SetStyle("gStyle");

	//gSystem->ProcessEvents();

}

#endif
