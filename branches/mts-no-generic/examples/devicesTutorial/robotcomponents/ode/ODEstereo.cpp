

// Need to include these first otherwise there's a mess with
//  #defines in cisstConfig.h
#include <osgViewer/CompositeViewer>
#include <osgGA/TrackballManipulator>

#include <cisstDevices/robotcomponents/osg/devOSGWorld.h>
#include <cisstDevices/robotcomponents/osg/devOSGBody.h>

#include <cisstDevices/robotcomponents/ode/devODEWorld.h>
#include <cisstDevices/robotcomponents/ode/devODEBody.h>
#include <cisstCommon/cmnGetChar.h>

#include <cisstMultiTask/mtsTaskManager.h>

#include <cisstCommon/cmnGetChar.h>

int main(){

  mtsTaskManager* taskManager = mtsTaskManager::GetInstance();

  devODEWorld* world;
  world = new devODEWorld( 0.001,
			   OSA_CPU1,
  			   vctFixedSizeVector<double,3>(0.0,0.0,-9.81) );
  taskManager->AddComponent( world );

  devODEBody* hubble;
  hubble = new devODEBody( "hubble",                              // name
			   vctFrame4x4<double>( vctMatrixRotation3<double>(),
						vctFixedSizeVector<double,3>(0, 0, 5.1 )),
			   1.0,                                   // mass
			   vctFixedSizeVector<double,3>( 0.0 ),   // com
			   vctFixedSizeMatrix<double,3,3>::Eye(), // moit
			   "libs/etc/cisstRobot/objects/hst.3ds", // model
			   world->GetSpaceID(),                   // 
			   world );
  devODEBody* background;
  background = new devODEBody( "background", 
			       vctFrame4x4<double>(),
			       "libs/etc/cisstRobot/objects/background.3ds",
			       world->GetSpaceID(),
			       world );

  taskManager->CreateAll();
  taskManager->StartAll();

  // Create a viewer
  osgViewer::CompositeViewer viewer;
  
  // Create the camera
  osg::Camera* leftcam = new osg::Camera();
  // set black background
  leftcam->setClearColor( osg::Vec4( 0.0, 0.0, 0.0, 1.0 ) );
  // set the projection matrix: viewing algle, ratio, near, far
  leftcam->setProjectionMatrixAsPerspective( 55, 512.0/480.0, 0.01, 10.0 );
  // set the position/orientation of the camera
  leftcam->setViewMatrixAsLookAt(  osg::Vec3d( 2, -0.1, 2 ),
				   osg::Vec3d( 0, -0.1, 0 ),
				   osg::Vec3d( 0,  0, 1 ) ); 

  // create a view for the left camera
  osgViewer::View* leftview = new osgViewer::View();
  leftview->setCamera( leftcam );
  leftview->setSceneData( world );
  leftview->setUpViewInWindow( 0, 0, 512, 480 );
  viewer.addView( leftview );

  // perform the same for the right camera
  osg::Camera* rightcam = new osg::Camera();
  rightcam->setClearColor( osg::Vec4( 0.0, 0.0, 0.0, 1.0 ) );
  rightcam->setProjectionMatrixAsPerspective( 55, 512.0/480.0, 0.01, 10.0 );
  rightcam->setViewMatrixAsLookAt(  osg::Vec3d( 2, 0.1, 2 ),
				    osg::Vec3d( 0, 0.1, 0 ),
				    osg::Vec3d( 0, 0, 1 ) ); 
  osgViewer::View* rightview = new osgViewer::View();
  rightview->setCamera( rightcam );
  rightview->setSceneData( world );
  rightview->setUpViewInWindow( 512, 0, 512, 480 );
  viewer.addView( rightview );    

  // run the viewer
  while(!viewer.done()){ 
    // Query the contacts for the hubble body
    std::list< devODEWorld::Contact > c = world->QueryContacts( "hubble" );
    std::list< devODEWorld::Contact >::const_iterator contact;
    for( contact=c.begin(); contact!=c.end(); contact++ )
      { std::cout << *contact << std::endl; }

    viewer.frame(); 
  }

  taskManager->KillAll();

  return 0;

}
