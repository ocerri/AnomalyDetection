#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>  // necessario per scrivere su file
#include <string>

#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TFile.h"
#include "TF1.h"
#include <TF2.h>
#include "TGraph.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TKey.h"
#include "TLegend.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include<TLine.h>

#include "TStopwatch.h"


#include "cebefo_style.h"


#ifndef __INCLUSE__
#define __INCLUSE__

Int_t segno(Int_t a)
{
	if (a>=0) return 1;
	else return -1;
}

Double_t reduce_angle(Double_t angle)
{
	Double_t pi = TMath::Pi();
	while(angle > pi) angle -= 2*pi;
  while(angle < -pi) angle += 2*pi;
	return angle;
}

TString GetTreeNameFromFile(TString filepath)
{
  TFile *f1 = TFile::Open(filepath);
  TIter keyList(f1->GetListOfKeys());
  TKey *key;
	TString name = "";
	Int_t found = 0;
  while ((key = (TKey*)keyList()) && !found) {
    TClass *cl = gROOT->GetClass(key->GetClassName());
    if (cl->InheritsFrom("TTree"))
    {
      name = key->GetTitle();
			found = 1;
    }
  }

	f1->Close();
	delete f1;

  return name;
}


using namespace std;
TStopwatch* watch_progress_bar;// = new TStopwatch();
Int_t st_time_progress_bar = 0;

void show_progress_bar(int entry, int numberOfEntries){
	Int_t step = numberOfEntries>200 ?  numberOfEntries/ 200 : 2;
	if (entry==0 || entry%step==0 || entry==numberOfEntries-1){
      if (entry==0){
				watch_progress_bar = new TStopwatch();
				watch_progress_bar->Start();
    		cout << endl << Form("[--------------------]   0%% ") << flush;
				st_time_progress_bar = time(nullptr);
      }
      else if((entry+1)/(float)numberOfEntries == 1){
				watch_progress_bar->Stop();
    		cout << '\r' << Form("[####################]  100%%                          ") << endl;
				cout <<"Elapsed time: " << watch_progress_bar->RealTime() << " sec (real time)"<< endl << endl;
				delete watch_progress_bar;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.95){
    		cout << '\r' << Form("[###################-]  95%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.90){
    		cout << '\r' << Form("[##################--]  90%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.85){
    		cout << '\r' << Form("[#################---]  85%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.80){
    		cout << '\r' << Form("[################----]  80%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.75){
    		cout << '\r' << Form("[###############-----]  75%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.70){
    		cout << '\r' << Form("[##############------]  70%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.65){
    		cout << '\r' << Form("[#############-------]  65%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.60){
    		cout << '\r' << Form("[############--------]  60%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.55){
    		cout << '\r' << Form("[###########---------]  55%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.50){
    		cout << '\r' << Form("[##########----------]  50%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.45){
    		cout << '\r' << Form("[#########-----------]  45%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.40){
    		cout << '\r' << Form("[########------------]  40%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.35){
    		cout << '\r' << Form("[#######-------------]  35%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.30){
    		cout << '\r' << Form("[######--------------]  30%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.25){
    		cout << '\r' << Form("[#####---------------]  25%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.20){
    		cout << '\r' << Form("[####----------------]  20%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.15){
    		cout << '\r' << Form("[###-----------------]  15%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.10){
    		cout << '\r' << Form("[##------------------]  10%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries >= 0.05){
    		cout << '\r' << Form("[#-------------------]   5%% ") << flush;
      }
      else if((entry+1)/(float)numberOfEntries < 0.05){
    		cout << '\r' << Form("[--------------------]   0%% ") << flush;
      }
    }
    cout << flush;

	if (entry>0 && entry%step==0)
	{
		Int_t now = time(nullptr);
		Double_t timeleft = (numberOfEntries - (Double_t)entry)*((Double_t)now - st_time_progress_bar)/(Double_t)entry;
		if(timeleft<181)
		{
			cout << Form(" -  remaning:%5.0f s   ",timeleft) << flush;
		}
		else if(timeleft<10801)
		{
			timeleft /= 60.;
			cout << Form(" -  remaning:%5.1f min ",timeleft) << flush;
		}
		else
		{
			timeleft /= 3600.;
			cout << Form(" -  remaning:%5.1f h   ",timeleft) << flush;
		}
	}
}

// Return the full width at half maximum of the given 1D histo
Double_t hist_FWHM(TH1F * h1)
{
	Int_t bin1 = h1->FindFirstBinAbove(h1->GetMaximum()/2);
  Int_t bin2 = h1->FindLastBinAbove(h1->GetMaximum()/2);
  return h1->GetBinCenter(bin2) - h1->GetBinCenter(bin1);
}

Double_t hist_FWHM(TH1D * h1)
{
	Int_t bin1 = h1->FindFirstBinAbove(h1->GetMaximum()/2);
 	Int_t bin2 = h1->FindLastBinAbove(h1->GetMaximum()/2);
 	return h1->GetBinCenter(bin2) - h1->GetBinCenter(bin1);
}

TCanvas* create_canvas(TString name)
{
	return new TCanvas(name,name,800,600);
}

Double_t performChi2Test(TH1D* h1, TH1D* h2)
{
	Int_t nbin = h1->GetNbinsX();
	if (h2->GetNbinsX() != nbin)
	{
		cout << "Error" << endl;
		return -1;
	}

	Double_t chi2 = 0;
	for(Int_t i=1; i<= nbin; i++)
	{
		Double_t aux = h1->GetBinContent(i) - h2->GetBinContent(i);
		chi2 += aux * aux;//*(err/sqrt(h1->GetBinContent(i) + h2->GetBinContent(i))
	}

	return chi2;
}

TCanvas* make_ratio_plot(TH1D* hin1, TH1D* hin2, TString title = "", Int_t Plot_stat_bound = 0, TString label = "")
{
  TH1D* h1 = (TH1D*) hin1->Clone("h1aux"+label);
  TH1D* h2 = (TH1D*) hin2->Clone("h2aux"+label);

  TString tag[] = {h1->GetTitle(), h2->GetTitle()};

  TCanvas *c_out = new TCanvas("c_out_ratio"+label, "c_out_ratio"+label, 600, 800);
  TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
  pad1->SetBottomMargin(0.03); // Upper and lower plot are joined
  pad1->SetLeftMargin(0.15); // Upper and lower plot are joined
  pad1->SetGridx();         // Vertical grid
  pad1->Draw();             // Draw the upper pad: pad1
  pad1->cd();               // pad1 becomes the current pad

  TLegend * leg = new TLegend(0.6, 0.7, 0.9, 0.9);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  c_out->cd(1);

  h1->GetXaxis()->SetLabelSize(0);
  h1->GetXaxis()->SetTitle("");

  h1->GetYaxis()->SetRangeUser(0, 1.05*max(h1->GetMaximum(), h2->GetMaximum()));
  h1->GetYaxis()->SetTitleOffset(1.5);
  h1->GetYaxis()->SetTitleSize(0.05);
  h1->GetYaxis()->SetLabelSize(0.05);
  leg->AddEntry(h1, tag[0], "lep");
  h1->SetTitle(title);
  h1->DrawCopy("E1");

  h2->DrawCopy("E1same");
  leg->AddEntry(h2, tag[1], "lep");

  leg->Draw("same");

  c_out->cd();          // Go back to the main canvas before defining pad2
  TPad *pad2 = new TPad("pad2", "pad2", 0, 0, 1, 0.3);
  pad2->SetTopMargin(0.03);
  pad2->SetBottomMargin(0.25);
  pad2->SetLeftMargin(0.15); // Upper and lower plot are joined
  pad2->SetGridx(); // vertical grid
  pad2->Draw();
  pad2->cd();       // pad2 becomes the current pad

  h2->Divide(h1);
  h2->GetYaxis()->SetTitleOffset(0.6);

  vector<Double_t> v;
	vector<Double_t> v_err;
	Double_t mean = 0;
	Double_t weights_sum = 0;
  for(Int_t i=1; i<h2->GetNbinsX()+1; i++)
  {
    if (h2->GetBinContent(i) != 0)
		{
			v.push_back(h2->GetBinContent(i));
			Double_t err2 = h2->GetBinError(i)*h2->GetBinError(i);
			v_err.push_back(err2);
			mean += h2->GetBinContent(i)/err2;
			weights_sum += 1/err2;
		}
  }
	mean /= weights_sum;
  Double_t RMS = 0;
	for(UInt_t i=0; i<v.size(); i++)
	{
		RMS += (mean - v[i])*(mean - v[i])/v_err[i];
	}
	RMS = sqrt(RMS/weights_sum);

	std::sort(v.begin(), v.end());
	Int_t i_min = (Int_t)(v.size()*0.05);
	Int_t i_max = (Int_t)(v.size()*0.95);

	Double_t upper_limit = max(mean + 4*RMS, Plot_stat_bound*1.1*v[i_max]);
	Double_t lower_limit = min(mean - 4*RMS, 0.9*v[i_min]);
	if (upper_limit>5) lower_limit = max(0.01, lower_limit);
	else lower_limit = max(0., lower_limit);


  h2->GetYaxis()->SetRangeUser(lower_limit, upper_limit);
  h2->GetYaxis()->SetTitleSize(0.12);
  h2->GetYaxis()->SetLabelSize(0.12);
  h2->GetYaxis()->SetNdivisions(506);
  h2->GetXaxis()->SetTitleOffset(0.95);
  h2->GetXaxis()->SetTitleSize(0.12);
  h2->GetXaxis()->SetLabelSize(0.12);
  h2->GetXaxis()->SetTickSize(0.07);
  h2->SetYTitle(Form("%s/%s", tag[1].Data(), tag[0].Data()));
  h2->SetTitle("");
  h2->DrawCopy("E1");
	if(upper_limit>5) pad2->SetLogy();
	pad2->Update();


  if(Plot_stat_bound)
  {
    for(Int_t i=-1; i<2; i++)
    {
      TLine* ln = new TLine(h2->GetXaxis()->GetXmin(), mean + i*RMS, h2->GetXaxis()->GetXmax(), mean + i*RMS);
      ln->SetLineWidth(1);
      ln->SetLineStyle(7);
      ln->SetLineColor(h2->GetLineColor());
      ln->Draw("SAME");
    }
  }

  TLine* ln = new TLine(h2->GetXaxis()->GetXmin(), 1, h2->GetXaxis()->GetXmax(), 1);
  ln->SetLineWidth(1);
  ln->SetLineColor(45);
  ln->Draw("SAME");

  return c_out;
}

TCanvas* make_ratio_plot(TH2D* hin1, TH2D* hin2, TString title = "", TString opt = "E", TString label = "")
{
	gStyle->SetPaintTextFormat("1.2f");

  TH2D* h1 = (TH2D*) hin1->Clone("h1aux"+label);
  TH2D* h2 = (TH2D*) hin2->Clone("h2aux"+label);

	h2->Divide(h1);

  TString tag[] = {h1->GetTitle(), h2->GetTitle()};

  TCanvas *c_out = new TCanvas("c_out_ratio"+label, "c_out_ratio"+label, 600, 800);
  h2->GetZaxis()->SetTitle(tag[1]+"/"+tag[0]);
	h2->GetZaxis()->SetRangeUser(0.5, 1.5);
	h2->SetTitle(title + Form(", corr: %.2f", h2->GetCorrelationFactor()));
	h2->SetStats(0);
	if(opt=="TEXT")
	{
		h2->Draw("COLZ");
		opt +="SAME";
	}
	h2->Draw(opt);

  return c_out;
}

TCanvas* make_significance_plot(TH2D* hin1, TH2D* hin2, TString title = "", TString opt = "COLZ", TString label = "")
{
	gStyle->SetPaintTextFormat("1.2f");

  TH2D* h1 = (TH2D*) hin1->Clone("h1aux"+label);
	TH2D* h2 = (TH2D*) hin2->Clone("h2aux"+label);
	h1->Scale(1./h1->Integral());
	h2->Scale(1./h2->Integral());

	TH2D* h_out = (TH2D*) hin2->Clone("h_out"+label);

	for(Int_t ix=1; ix <= h1->GetNbinsX(); ix++)
	{
		for(Int_t iy=1; iy <= h1->GetNbinsY(); iy++)
		{
			Double_t n1 = h1->GetBinContent(ix, iy);
			Double_t n2 = h2->GetBinContent(ix, iy);

			Double_t d1 = h1->GetBinError(ix, iy);
			Double_t d2 = h2->GetBinError(ix, iy);

			Double_t aux = 0;
			if(n1*n2 != 0)
			{
				aux = (n2/n1 - 1)/(sqrt(pow(d1/n1,2) + pow(d2/n2,2))*n2/n1);
				// cout << n2/n1 << " " << aux << endl;
			}

			h_out->SetBinContent(ix, iy, aux);
		}
	}


  TString tag[] = {h1->GetTitle(), h2->GetTitle()};

  TCanvas *c_out = new TCanvas("c_out_significance"+label, "c_out_significance"+label, 600, 800);
  h_out->GetZaxis()->SetTitle("Significance of "+tag[1]+"/"+tag[0]);
	h_out->SetTitle(title + Form(", corr: %.2f", h_out->GetCorrelationFactor()));
	h_out->SetStats(0);
	if(opt=="TEXT")
	{
		h_out->Draw("COLZ");
		opt +="SAME";
	}
	h_out->Draw(opt);

  return c_out;
}

void incluse()
{

}

void incluse_lxplus()
{

}

#endif
