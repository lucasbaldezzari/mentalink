<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE eagle SYSTEM "eagle.dtd">
<eagle version="9.6.2">
<drawing>
<settings>
<setting alwaysvectorfont="no"/>
<setting verticaltext="up"/>
</settings>
<grid distance="0.1" unitdist="inch" unit="inch" style="lines" multiple="1" display="no" altdistance="0.01" altunitdist="inch" altunit="inch"/>
<layers>
<layer number="1" name="Top" color="4" fill="1" visible="no" active="no"/>
<layer number="2" name="Route2" color="1" fill="3" visible="no" active="no"/>
<layer number="3" name="Route3" color="4" fill="3" visible="no" active="no"/>
<layer number="4" name="Route4" color="1" fill="4" visible="no" active="no"/>
<layer number="5" name="Route5" color="4" fill="4" visible="no" active="no"/>
<layer number="6" name="Route6" color="1" fill="8" visible="no" active="no"/>
<layer number="7" name="Route7" color="4" fill="8" visible="no" active="no"/>
<layer number="8" name="Route8" color="1" fill="2" visible="no" active="no"/>
<layer number="9" name="Route9" color="4" fill="2" visible="no" active="no"/>
<layer number="10" name="Route10" color="1" fill="7" visible="no" active="no"/>
<layer number="11" name="Route11" color="4" fill="7" visible="no" active="no"/>
<layer number="12" name="Route12" color="1" fill="5" visible="no" active="no"/>
<layer number="13" name="Route13" color="4" fill="5" visible="no" active="no"/>
<layer number="14" name="Route14" color="1" fill="6" visible="no" active="no"/>
<layer number="15" name="Route15" color="4" fill="6" visible="no" active="no"/>
<layer number="16" name="Bottom" color="1" fill="1" visible="no" active="no"/>
<layer number="17" name="Pads" color="2" fill="1" visible="no" active="no"/>
<layer number="18" name="Vias" color="2" fill="1" visible="no" active="no"/>
<layer number="19" name="Unrouted" color="6" fill="1" visible="no" active="no"/>
<layer number="20" name="Dimension" color="15" fill="1" visible="no" active="no"/>
<layer number="21" name="tPlace" color="7" fill="1" visible="no" active="no"/>
<layer number="22" name="bPlace" color="7" fill="1" visible="no" active="no"/>
<layer number="23" name="tOrigins" color="15" fill="1" visible="no" active="no"/>
<layer number="24" name="bOrigins" color="15" fill="1" visible="no" active="no"/>
<layer number="25" name="tNames" color="7" fill="1" visible="no" active="no"/>
<layer number="26" name="bNames" color="7" fill="1" visible="no" active="no"/>
<layer number="27" name="tValues" color="7" fill="1" visible="no" active="no"/>
<layer number="28" name="bValues" color="7" fill="1" visible="no" active="no"/>
<layer number="29" name="tStop" color="7" fill="3" visible="no" active="no"/>
<layer number="30" name="bStop" color="7" fill="6" visible="no" active="no"/>
<layer number="31" name="tCream" color="7" fill="4" visible="no" active="no"/>
<layer number="32" name="bCream" color="7" fill="5" visible="no" active="no"/>
<layer number="33" name="tFinish" color="6" fill="3" visible="no" active="no"/>
<layer number="34" name="bFinish" color="6" fill="6" visible="no" active="no"/>
<layer number="35" name="tGlue" color="7" fill="4" visible="no" active="no"/>
<layer number="36" name="bGlue" color="7" fill="5" visible="no" active="no"/>
<layer number="37" name="tTest" color="7" fill="1" visible="no" active="no"/>
<layer number="38" name="bTest" color="7" fill="1" visible="no" active="no"/>
<layer number="39" name="tKeepout" color="4" fill="11" visible="no" active="no"/>
<layer number="40" name="bKeepout" color="1" fill="11" visible="no" active="no"/>
<layer number="41" name="tRestrict" color="4" fill="10" visible="no" active="no"/>
<layer number="42" name="bRestrict" color="1" fill="10" visible="no" active="no"/>
<layer number="43" name="vRestrict" color="2" fill="10" visible="no" active="no"/>
<layer number="44" name="Drills" color="7" fill="1" visible="no" active="no"/>
<layer number="45" name="Holes" color="7" fill="1" visible="no" active="no"/>
<layer number="46" name="Milling" color="3" fill="1" visible="no" active="no"/>
<layer number="47" name="Measures" color="7" fill="1" visible="no" active="no"/>
<layer number="48" name="Document" color="7" fill="1" visible="no" active="no"/>
<layer number="49" name="Reference" color="7" fill="1" visible="no" active="no"/>
<layer number="51" name="tDocu" color="7" fill="1" visible="no" active="no"/>
<layer number="52" name="bDocu" color="7" fill="1" visible="no" active="no"/>
<layer number="88" name="SimResults" color="9" fill="1" visible="yes" active="yes"/>
<layer number="89" name="SimProbes" color="9" fill="1" visible="yes" active="yes"/>
<layer number="90" name="Modules" color="5" fill="1" visible="yes" active="yes"/>
<layer number="91" name="Nets" color="2" fill="1" visible="yes" active="yes"/>
<layer number="92" name="Busses" color="1" fill="1" visible="yes" active="yes"/>
<layer number="93" name="Pins" color="2" fill="1" visible="yes" active="yes"/>
<layer number="94" name="Symbols" color="4" fill="1" visible="yes" active="yes"/>
<layer number="95" name="Names" color="7" fill="1" visible="yes" active="yes"/>
<layer number="96" name="Values" color="7" fill="1" visible="yes" active="yes"/>
<layer number="97" name="Info" color="7" fill="1" visible="yes" active="yes"/>
<layer number="98" name="Guide" color="6" fill="1" visible="yes" active="yes"/>
<layer number="99" name="SpiceOrder" color="7" fill="1" visible="yes" active="yes"/>
<layer number="100" name="Busses2" color="7" fill="1" visible="yes" active="yes"/>
</layers>
<schematic xreflabel="%F%N/%S.%C%R" xrefpart="/%S.%C%R">
<libraries>
<library name="st-microelectronics" urn="urn:adsk.eagle:library:368">
<description>&lt;b&gt;ST Microelectronics Devices&lt;/b&gt;&lt;p&gt;
Microcontrollers,  I2C components, linear devices&lt;p&gt;
http://www.st.com&lt;p&gt;
&lt;i&gt;Include st-microelectronics-2.lbr, st-microelectronics-3.lbr.&lt;p&gt;&lt;/i&gt;

&lt;author&gt;Created by librarian@cadsoft.de&lt;/author&gt;</description>
<packages>
<package name="DIL16" urn="urn:adsk.eagle:footprint:26704/1" library_version="6">
<description>&lt;b&gt;Dual In Line Package&lt;/b&gt;</description>
<wire x1="10.16" y1="2.921" x2="-10.16" y2="2.921" width="0.1524" layer="21"/>
<wire x1="-10.16" y1="-2.921" x2="10.16" y2="-2.921" width="0.1524" layer="21"/>
<wire x1="10.16" y1="2.921" x2="10.16" y2="-2.921" width="0.1524" layer="21"/>
<wire x1="-10.16" y1="2.921" x2="-10.16" y2="1.016" width="0.1524" layer="21"/>
<wire x1="-10.16" y1="-2.921" x2="-10.16" y2="-1.016" width="0.1524" layer="21"/>
<wire x1="-10.16" y1="1.016" x2="-10.16" y2="-1.016" width="0.1524" layer="21" curve="-180"/>
<pad name="1" x="-8.89" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="2" x="-6.35" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="7" x="6.35" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="8" x="8.89" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="3" x="-3.81" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="4" x="-1.27" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="6" x="3.81" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="5" x="1.27" y="-3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="9" x="8.89" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="10" x="6.35" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="11" x="3.81" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="12" x="1.27" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="13" x="-1.27" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="14" x="-3.81" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="15" x="-6.35" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<pad name="16" x="-8.89" y="3.81" drill="0.8128" shape="long" rot="R90"/>
<text x="-10.541" y="-2.921" size="1.27" layer="25" ratio="10" rot="R90">&gt;NAME</text>
<text x="-7.493" y="-0.635" size="1.27" layer="27" ratio="10">&gt;VALUE</text>
</package>
</packages>
<packages3d>
<package3d name="DIL16" urn="urn:adsk.eagle:package:26820/1" type="box" library_version="6">
<description>Dual In Line Package</description>
<packageinstances>
<packageinstance name="DIL16"/>
</packageinstances>
</package3d>
</packages3d>
<symbols>
<symbol name="L293D" urn="urn:adsk.eagle:symbol:26701/1" library_version="6">
<wire x1="-10.16" y1="20.32" x2="-10.16" y2="-20.32" width="0.254" layer="94"/>
<wire x1="-10.16" y1="-20.32" x2="10.16" y2="-20.32" width="0.254" layer="94"/>
<wire x1="10.16" y1="-20.32" x2="10.16" y2="20.32" width="0.254" layer="94"/>
<wire x1="10.16" y1="20.32" x2="-10.16" y2="20.32" width="0.254" layer="94"/>
<text x="-10.16" y="21.336" size="1.778" layer="95">&gt;NAME</text>
<text x="-10.16" y="-22.86" size="1.778" layer="96">&gt;VALUE</text>
<pin name="1-2EN" x="-15.24" y="17.78" length="middle" direction="in"/>
<pin name="1A" x="-15.24" y="12.7" length="middle" direction="in"/>
<pin name="1Y" x="-15.24" y="7.62" length="middle" direction="out"/>
<pin name="GND1" x="-15.24" y="2.54" length="middle" direction="pwr"/>
<pin name="GND2" x="-15.24" y="-2.54" length="middle" direction="pwr"/>
<pin name="2Y" x="-15.24" y="-7.62" length="middle" direction="out"/>
<pin name="2A" x="-15.24" y="-12.7" length="middle" direction="in"/>
<pin name="VCC2" x="-15.24" y="-17.78" length="middle" direction="pwr"/>
<pin name="VCC1" x="15.24" y="17.78" length="middle" direction="pwr" rot="R180"/>
<pin name="4A" x="15.24" y="12.7" length="middle" direction="in" rot="R180"/>
<pin name="4Y" x="15.24" y="7.62" length="middle" direction="out" rot="R180"/>
<pin name="GND3" x="15.24" y="2.54" length="middle" direction="pwr" rot="R180"/>
<pin name="GND4" x="15.24" y="-2.54" length="middle" direction="pwr" rot="R180"/>
<pin name="3Y" x="15.24" y="-7.62" length="middle" direction="out" rot="R180"/>
<pin name="3A" x="15.24" y="-12.7" length="middle" direction="in" rot="R180"/>
<pin name="3-4EN" x="15.24" y="-17.78" length="middle" direction="in" rot="R180"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="L293D" urn="urn:adsk.eagle:component:26880/3" prefix="IC" library_version="6">
<description>&lt;b&gt;PUSH-PULL 4 CHANNEL DRIVER&lt;/b&gt;</description>
<gates>
<gate name="G$1" symbol="L293D" x="0" y="0"/>
</gates>
<devices>
<device name="" package="DIL16">
<connects>
<connect gate="G$1" pin="1-2EN" pad="1"/>
<connect gate="G$1" pin="1A" pad="2"/>
<connect gate="G$1" pin="1Y" pad="3"/>
<connect gate="G$1" pin="2A" pad="7"/>
<connect gate="G$1" pin="2Y" pad="6"/>
<connect gate="G$1" pin="3-4EN" pad="9"/>
<connect gate="G$1" pin="3A" pad="10"/>
<connect gate="G$1" pin="3Y" pad="11"/>
<connect gate="G$1" pin="4A" pad="15"/>
<connect gate="G$1" pin="4Y" pad="14"/>
<connect gate="G$1" pin="GND1" pad="4"/>
<connect gate="G$1" pin="GND2" pad="5"/>
<connect gate="G$1" pin="GND3" pad="13"/>
<connect gate="G$1" pin="GND4" pad="12"/>
<connect gate="G$1" pin="VCC1" pad="16"/>
<connect gate="G$1" pin="VCC2" pad="8"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:26820/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="MF" value="" constant="no"/>
<attribute name="MPN" value="L293D" constant="no"/>
<attribute name="OC_FARNELL" value="9589619" constant="no"/>
<attribute name="OC_NEWARK" value="56P8249" constant="no"/>
<attribute name="POPULARITY" value="16" constant="no"/>
</technology>
</technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="con-molex" urn="urn:adsk.eagle:library:165">
<description>&lt;b&gt;Molex Connectors&lt;/b&gt;&lt;p&gt;
&lt;author&gt;Created by librarian@cadsoft.de&lt;/author&gt;</description>
<packages>
<package name="22-23-2021" urn="urn:adsk.eagle:footprint:8078259/1" library_version="5">
<description>&lt;b&gt;KK® 254 Solid Header, Vertical, with Friction Lock, 2 Circuits, Tin (Sn) Plating&lt;/b&gt;&lt;p&gt;&lt;a href =http://www.molex.com/pdm_docs/sd/022232021_sd.pdf&gt;Datasheet &lt;/a&gt;</description>
<wire x1="-2.54" y1="3.175" x2="2.54" y2="3.175" width="0.254" layer="21"/>
<wire x1="2.54" y1="3.175" x2="2.54" y2="1.27" width="0.254" layer="21"/>
<wire x1="2.54" y1="1.27" x2="2.54" y2="-3.175" width="0.254" layer="21"/>
<wire x1="2.54" y1="-3.175" x2="-2.54" y2="-3.175" width="0.254" layer="21"/>
<wire x1="-2.54" y1="-3.175" x2="-2.54" y2="1.27" width="0.254" layer="21"/>
<wire x1="-2.54" y1="1.27" x2="-2.54" y2="3.175" width="0.254" layer="21"/>
<wire x1="-2.54" y1="1.27" x2="2.54" y2="1.27" width="0.254" layer="21"/>
<pad name="1" x="-1.27" y="0" drill="1" shape="long" rot="R90"/>
<pad name="2" x="1.27" y="0" drill="1" shape="long" rot="R90"/>
<text x="-2.54" y="3.81" size="1.016" layer="25" ratio="10">&gt;NAME</text>
<text x="-2.54" y="-5.08" size="1.016" layer="27" ratio="10">&gt;VALUE</text>
</package>
</packages>
<packages3d>
<package3d name="22-23-2021" urn="urn:adsk.eagle:package:8078633/1" type="box" library_version="5">
<description>&lt;b&gt;KK® 254 Solid Header, Vertical, with Friction Lock, 2 Circuits, Tin (Sn) Plating&lt;/b&gt;&lt;p&gt;&lt;a href =http://www.molex.com/pdm_docs/sd/022232021_sd.pdf&gt;Datasheet &lt;/a&gt;</description>
<packageinstances>
<packageinstance name="22-23-2021"/>
</packageinstances>
</package3d>
</packages3d>
<symbols>
<symbol name="MV" urn="urn:adsk.eagle:symbol:6783/2" library_version="5">
<wire x1="1.27" y1="0" x2="0" y2="0" width="0.6096" layer="94"/>
<text x="2.54" y="-0.762" size="1.524" layer="95">&gt;NAME</text>
<text x="-0.762" y="1.397" size="1.778" layer="96">&gt;VALUE</text>
<pin name="S" x="-2.54" y="0" visible="off" length="short" direction="pas"/>
</symbol>
<symbol name="M" urn="urn:adsk.eagle:symbol:6785/2" library_version="5">
<wire x1="1.27" y1="0" x2="0" y2="0" width="0.6096" layer="94"/>
<text x="2.54" y="-0.762" size="1.524" layer="95">&gt;NAME</text>
<pin name="S" x="-2.54" y="0" visible="off" length="short" direction="pas"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="22-23-2021" urn="urn:adsk.eagle:component:8078938/3" prefix="X" library_version="5">
<description>.100" (2.54mm) Center Header - 2 Pin</description>
<gates>
<gate name="-1" symbol="MV" x="0" y="0" addlevel="always" swaplevel="1"/>
<gate name="-2" symbol="M" x="0" y="-2.54" addlevel="always" swaplevel="1"/>
</gates>
<devices>
<device name="" package="22-23-2021">
<connects>
<connect gate="-1" pin="S" pad="1"/>
<connect gate="-2" pin="S" pad="2"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:8078633/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="MF" value="MOLEX" constant="no"/>
<attribute name="MPN" value="22-23-2021" constant="no"/>
<attribute name="OC_FARNELL" value="1462926" constant="no"/>
<attribute name="OC_NEWARK" value="25C3832" constant="no"/>
<attribute name="POPULARITY" value="40" constant="no"/>
</technology>
</technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
</libraries>
<attributes>
</attributes>
<variantdefs>
</variantdefs>
<classes>
<class number="0" name="default" width="0" drill="0">
</class>
</classes>
<parts>
<part name="IC1" library="st-microelectronics" library_urn="urn:adsk.eagle:library:368" deviceset="L293D" device="" package3d_urn="urn:adsk.eagle:package:26820/1"/>
<part name="IC2" library="st-microelectronics" library_urn="urn:adsk.eagle:library:368" deviceset="L293D" device="" package3d_urn="urn:adsk.eagle:package:26820/1"/>
<part name="12V" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="ARDUINO" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X3" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X4" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X5" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X6" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X7" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X8" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X9" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
<part name="X10" library="con-molex" library_urn="urn:adsk.eagle:library:165" deviceset="22-23-2021" device="" package3d_urn="urn:adsk.eagle:package:8078633/1"/>
</parts>
<sheets>
<sheet>
<plain>
</plain>
<instances>
<instance part="IC1" gate="G$1" x="35.56" y="78.74" smashed="yes" rot="R90"/>
<instance part="IC2" gate="G$1" x="86.36" y="78.74" smashed="yes" rot="R90"/>
<instance part="12V" gate="-1" x="-7.62" y="101.6" smashed="yes" rot="R180">
<attribute name="NAME" x="-10.16" y="102.362" size="1.524" layer="95" rot="R180"/>
</instance>
<instance part="12V" gate="-2" x="-7.62" y="99.06" smashed="yes" rot="R180">
<attribute name="NAME" x="-10.16" y="99.822" size="1.524" layer="95" rot="R180"/>
</instance>
<instance part="ARDUINO" gate="-1" x="-10.16" y="55.88" smashed="yes" rot="R180">
<attribute name="NAME" x="-15.24" y="56.642" size="1.524" layer="95" rot="R180"/>
</instance>
<instance part="ARDUINO" gate="-2" x="-10.16" y="53.34" smashed="yes" rot="R180">
<attribute name="NAME" x="-15.24" y="54.102" size="1.524" layer="95" rot="R180"/>
</instance>
<instance part="X3" gate="-1" x="27.94" y="40.64" smashed="yes" rot="R270">
<attribute name="NAME" x="27.178" y="38.1" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X3" gate="-2" x="22.86" y="40.64" smashed="yes" rot="R270">
<attribute name="NAME" x="22.098" y="38.1" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X4" gate="-1" x="48.26" y="40.64" smashed="yes" rot="R270">
<attribute name="NAME" x="47.498" y="38.1" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X4" gate="-2" x="43.18" y="40.64" smashed="yes" rot="R270">
<attribute name="NAME" x="42.418" y="38.1" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X5" gate="-1" x="78.74" y="38.1" smashed="yes" rot="R270">
<attribute name="NAME" x="77.978" y="35.56" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X5" gate="-2" x="73.66" y="38.1" smashed="yes" rot="R270">
<attribute name="NAME" x="72.898" y="35.56" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X6" gate="-1" x="99.06" y="40.64" smashed="yes" rot="R270">
<attribute name="NAME" x="98.298" y="38.1" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X6" gate="-2" x="93.98" y="40.64" smashed="yes" rot="R270">
<attribute name="NAME" x="93.218" y="38.1" size="1.524" layer="95" rot="R270"/>
</instance>
<instance part="X7" gate="-1" x="93.98" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="94.742" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X7" gate="-2" x="99.06" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="97.282" y="116.84" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X8" gate="-1" x="73.66" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="74.422" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X8" gate="-2" x="78.74" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="79.502" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X9" gate="-1" x="43.18" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="43.942" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X9" gate="-2" x="48.26" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="49.022" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X10" gate="-1" x="22.86" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="23.622" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
<instance part="X10" gate="-2" x="27.94" y="116.84" smashed="yes" rot="R90">
<attribute name="NAME" x="28.702" y="119.38" size="1.524" layer="95" rot="R90"/>
</instance>
</instances>
<busses>
</busses>
<nets>
<net name="N$1" class="0">
<segment>
<pinref part="12V" gate="-2" pin="S"/>
<wire x1="-5.08" y1="99.06" x2="-5.08" y2="81.28" width="0.1524" layer="91"/>
<wire x1="-5.08" y1="81.28" x2="33.02" y2="81.28" width="0.1524" layer="91"/>
<pinref part="IC1" gate="G$1" pin="GND3"/>
<wire x1="33.02" y1="81.28" x2="33.02" y2="93.98" width="0.1524" layer="91"/>
<wire x1="33.02" y1="81.28" x2="38.1" y2="81.28" width="0.1524" layer="91"/>
<junction x="33.02" y="81.28"/>
<pinref part="IC1" gate="G$1" pin="GND4"/>
<wire x1="38.1" y1="81.28" x2="38.1" y2="93.98" width="0.1524" layer="91"/>
<pinref part="IC1" gate="G$1" pin="GND2"/>
<wire x1="38.1" y1="81.28" x2="38.1" y2="73.66" width="0.1524" layer="91"/>
<junction x="38.1" y="81.28"/>
<pinref part="IC1" gate="G$1" pin="GND1"/>
<wire x1="38.1" y1="73.66" x2="38.1" y2="63.5" width="0.1524" layer="91"/>
<wire x1="33.02" y1="81.28" x2="33.02" y2="73.66" width="0.1524" layer="91"/>
<wire x1="33.02" y1="73.66" x2="33.02" y2="63.5" width="0.1524" layer="91"/>
<wire x1="38.1" y1="81.28" x2="83.82" y2="81.28" width="0.1524" layer="91"/>
<pinref part="IC2" gate="G$1" pin="GND3"/>
<wire x1="83.82" y1="81.28" x2="83.82" y2="93.98" width="0.1524" layer="91"/>
<wire x1="83.82" y1="81.28" x2="88.9" y2="81.28" width="0.1524" layer="91"/>
<junction x="83.82" y="81.28"/>
<pinref part="IC2" gate="G$1" pin="GND4"/>
<wire x1="88.9" y1="93.98" x2="88.9" y2="81.28" width="0.1524" layer="91"/>
<pinref part="IC2" gate="G$1" pin="GND2"/>
<wire x1="88.9" y1="81.28" x2="88.9" y2="73.66" width="0.1524" layer="91"/>
<junction x="88.9" y="81.28"/>
<pinref part="IC2" gate="G$1" pin="GND1"/>
<wire x1="88.9" y1="73.66" x2="88.9" y2="63.5" width="0.1524" layer="91"/>
<wire x1="83.82" y1="81.28" x2="83.82" y2="73.66" width="0.1524" layer="91"/>
<pinref part="ARDUINO" gate="-1" pin="S"/>
<wire x1="83.82" y1="73.66" x2="83.82" y2="63.5" width="0.1524" layer="91"/>
<wire x1="-7.62" y1="55.88" x2="-5.08" y2="55.88" width="0.1524" layer="91"/>
<wire x1="-5.08" y1="55.88" x2="-5.08" y2="73.66" width="0.1524" layer="91"/>
<wire x1="-5.08" y1="73.66" x2="33.02" y2="73.66" width="0.1524" layer="91"/>
<wire x1="38.1" y1="73.66" x2="33.02" y2="73.66" width="0.1524" layer="91"/>
<junction x="38.1" y="73.66"/>
<junction x="33.02" y="73.66"/>
<wire x1="38.1" y1="73.66" x2="83.82" y2="73.66" width="0.1524" layer="91"/>
<junction x="83.82" y="73.66"/>
<wire x1="83.82" y1="73.66" x2="88.9" y2="73.66" width="0.1524" layer="91"/>
<junction x="88.9" y="73.66"/>
</segment>
</net>
<net name="N$3" class="0">
<segment>
<pinref part="IC2" gate="G$1" pin="4A"/>
<pinref part="X8" gate="-1" pin="S"/>
<wire x1="73.66" y1="93.98" x2="73.66" y2="114.3" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$4" class="0">
<segment>
<pinref part="IC2" gate="G$1" pin="4Y"/>
<pinref part="X8" gate="-2" pin="S"/>
<wire x1="78.74" y1="93.98" x2="78.74" y2="114.3" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$5" class="0">
<segment>
<pinref part="X7" gate="-1" pin="S"/>
<pinref part="IC2" gate="G$1" pin="3Y"/>
<wire x1="93.98" y1="114.3" x2="93.98" y2="93.98" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$6" class="0">
<segment>
<pinref part="X7" gate="-2" pin="S"/>
<pinref part="IC2" gate="G$1" pin="3A"/>
<wire x1="99.06" y1="114.3" x2="99.06" y2="93.98" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$7" class="0">
<segment>
<pinref part="X6" gate="-1" pin="S"/>
<pinref part="IC2" gate="G$1" pin="2A"/>
<wire x1="99.06" y1="43.18" x2="99.06" y2="63.5" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$8" class="0">
<segment>
<pinref part="X6" gate="-2" pin="S"/>
<pinref part="IC2" gate="G$1" pin="2Y"/>
<wire x1="93.98" y1="43.18" x2="93.98" y2="63.5" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$9" class="0">
<segment>
<pinref part="IC2" gate="G$1" pin="1Y"/>
<pinref part="X5" gate="-1" pin="S"/>
<wire x1="78.74" y1="63.5" x2="78.74" y2="40.64" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$10" class="0">
<segment>
<pinref part="IC2" gate="G$1" pin="1A"/>
<pinref part="X5" gate="-2" pin="S"/>
<wire x1="73.66" y1="63.5" x2="73.66" y2="40.64" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$11" class="0">
<segment>
<pinref part="IC1" gate="G$1" pin="2A"/>
<pinref part="X4" gate="-1" pin="S"/>
<wire x1="48.26" y1="63.5" x2="48.26" y2="43.18" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$12" class="0">
<segment>
<pinref part="IC1" gate="G$1" pin="2Y"/>
<pinref part="X4" gate="-2" pin="S"/>
<wire x1="43.18" y1="63.5" x2="43.18" y2="43.18" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$13" class="0">
<segment>
<pinref part="X3" gate="-1" pin="S"/>
<pinref part="IC1" gate="G$1" pin="1Y"/>
<wire x1="27.94" y1="43.18" x2="27.94" y2="63.5" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$14" class="0">
<segment>
<pinref part="X3" gate="-2" pin="S"/>
<pinref part="IC1" gate="G$1" pin="1A"/>
<wire x1="22.86" y1="43.18" x2="22.86" y2="63.5" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$15" class="0">
<segment>
<pinref part="X10" gate="-1" pin="S"/>
<pinref part="IC1" gate="G$1" pin="4A"/>
<wire x1="22.86" y1="114.3" x2="22.86" y2="93.98" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$16" class="0">
<segment>
<pinref part="X10" gate="-2" pin="S"/>
<pinref part="IC1" gate="G$1" pin="4Y"/>
<wire x1="27.94" y1="114.3" x2="27.94" y2="93.98" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$17" class="0">
<segment>
<pinref part="X9" gate="-1" pin="S"/>
<pinref part="IC1" gate="G$1" pin="3Y"/>
<wire x1="43.18" y1="114.3" x2="43.18" y2="93.98" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$18" class="0">
<segment>
<pinref part="X9" gate="-2" pin="S"/>
<pinref part="IC1" gate="G$1" pin="3A"/>
<wire x1="48.26" y1="114.3" x2="48.26" y2="93.98" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$19" class="0">
<segment>
<pinref part="IC1" gate="G$1" pin="VCC1"/>
<wire x1="17.78" y1="93.98" x2="17.78" y2="99.06" width="0.1524" layer="91"/>
<wire x1="17.78" y1="99.06" x2="68.58" y2="99.06" width="0.1524" layer="91"/>
<pinref part="IC2" gate="G$1" pin="VCC1"/>
<wire x1="68.58" y1="99.06" x2="68.58" y2="93.98" width="0.1524" layer="91"/>
<wire x1="68.58" y1="99.06" x2="111.76" y2="99.06" width="0.1524" layer="91"/>
<junction x="68.58" y="99.06"/>
<pinref part="IC2" gate="G$1" pin="VCC2"/>
<wire x1="111.76" y1="99.06" x2="111.76" y2="63.5" width="0.1524" layer="91"/>
<wire x1="111.76" y1="63.5" x2="104.14" y2="63.5" width="0.1524" layer="91"/>
<wire x1="104.14" y1="63.5" x2="104.14" y2="53.34" width="0.1524" layer="91"/>
<wire x1="104.14" y1="53.34" x2="53.34" y2="53.34" width="0.1524" layer="91"/>
<junction x="104.14" y="63.5"/>
<pinref part="IC1" gate="G$1" pin="VCC2"/>
<wire x1="53.34" y1="53.34" x2="53.34" y2="63.5" width="0.1524" layer="91"/>
<pinref part="ARDUINO" gate="-2" pin="S"/>
<wire x1="53.34" y1="53.34" x2="-7.62" y2="53.34" width="0.1524" layer="91"/>
<junction x="53.34" y="53.34"/>
</segment>
</net>
<net name="N$20" class="0">
<segment>
<pinref part="12V" gate="-1" pin="S"/>
<wire x1="-5.08" y1="101.6" x2="7.62" y2="101.6" width="0.1524" layer="91"/>
<wire x1="7.62" y1="101.6" x2="7.62" y2="63.5" width="0.1524" layer="91"/>
<pinref part="IC1" gate="G$1" pin="1-2EN"/>
<wire x1="7.62" y1="63.5" x2="17.78" y2="63.5" width="0.1524" layer="91"/>
<wire x1="17.78" y1="63.5" x2="17.78" y2="76.2" width="0.1524" layer="91"/>
<junction x="17.78" y="63.5"/>
<wire x1="17.78" y1="76.2" x2="53.34" y2="76.2" width="0.1524" layer="91"/>
<pinref part="IC1" gate="G$1" pin="3-4EN"/>
<wire x1="53.34" y1="76.2" x2="53.34" y2="93.98" width="0.1524" layer="91"/>
<wire x1="53.34" y1="76.2" x2="68.58" y2="76.2" width="0.1524" layer="91"/>
<junction x="53.34" y="76.2"/>
<pinref part="IC2" gate="G$1" pin="1-2EN"/>
<wire x1="68.58" y1="76.2" x2="68.58" y2="63.5" width="0.1524" layer="91"/>
<wire x1="68.58" y1="76.2" x2="104.14" y2="76.2" width="0.1524" layer="91"/>
<junction x="68.58" y="76.2"/>
<pinref part="IC2" gate="G$1" pin="3-4EN"/>
<wire x1="104.14" y1="93.98" x2="104.14" y2="76.2" width="0.1524" layer="91"/>
</segment>
</net>
</nets>
</sheet>
</sheets>
</schematic>
</drawing>
<compatibility>
<note version="8.2" severity="warning">
Since Version 8.2, EAGLE supports online libraries. The ids
of those online libraries will not be understood (or retained)
with this version.
</note>
<note version="8.3" severity="warning">
Since Version 8.3, EAGLE supports URNs for individual library
assets (packages, symbols, and devices). The URNs of those assets
will not be understood (or retained) with this version.
</note>
<note version="8.3" severity="warning">
Since Version 8.3, EAGLE supports the association of 3D packages
with devices in libraries, schematics, and board files. Those 3D
packages will not be understood (or retained) with this version.
</note>
</compatibility>
</eagle>
