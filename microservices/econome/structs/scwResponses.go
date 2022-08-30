package scwResponses

type ScwServerResponse struct {
	Server Server `json:"server"`
}

type Server struct {
	ID                string         	`json:"id"`                 
	Name              string         	`json:"name"`               
	Organization      string         	`json:"organization"`       
	Project           string         	`json:"project"`            
	AllowedActions    []string       	`json:"allowed_actions"`    
	Tags              []string       	`json:"tags"`               
	CommercialType    string         	`json:"commercial_type"`    
	CreationDate      string         	`json:"creation_date"`      
	DynamicIPRequired string         	`json:"dynamic_ip_required"`
	EnableIpv6        string         	`json:"enable_ipv6"`        
	Hostname          string         	`json:"hostname"`           
	Image             Image          	`json:"image"`              
	Protected         string         	`json:"protected"`          
	PrivateIP         string         	`json:"private_ip"`         
	PublicIP          PublicIP       	`json:"public_ip"`          
	ModificationDate  string         	`json:"modification_date"`  
	State             string         	`json:"state"`              
	Location          Location       	`json:"location"`           
	Ipv6              Ipv6           	`json:"ipv6"`               
	Bootscript        Bootscript     	`json:"bootscript"`         
	BootType          string         	`json:"boot_type"`          
	Volumes           map[string]Volume	`json:"volumes"`            
	SecurityGroup     SecurityGroup  	`json:"security_group"`     
	Maintenances      []Maintenance  	`json:"maintenances"`       
	StateDetail       string         	`json:"state_detail"`       
	Arch              string         	`json:"arch"`               
	PlacementGroup    PlacementGroup 	`json:"placement_group"`    
	PrivateNics       []PrivateNIC   	`json:"private_nics"`       
	Zone              string         	`json:"zone"`               
}

type Bootscript struct {
	Bootcmdargs  string `json:"bootcmdargs"` 
	Default      string `json:"default"`     
	Dtb          string `json:"dtb"`         
	ID           string `json:"id"`          
	Initrd       string `json:"initrd"`      
	Kernel       string `json:"kernel"`      
	Organization string `json:"organization"`
	Project      string `json:"project"`     
	Public       string `json:"public"`      
	Title        string `json:"title"`       
	Arch         string `json:"arch"`        
	Zone         string `json:"zone"`        
}

type Image struct {
	ID                string       		`json:"id"`                
	Name              string       		`json:"name"`              
	Arch              string       		`json:"arch"`              
	CreationDate      string       		`json:"creation_date"`     
	ModificationDate  string       		`json:"modification_date"` 
	DefaultBootscript Bootscript   		`json:"default_bootscript"`
	ExtraVolumes      map[string]Volume `json:"extra_volumes"`     
	FromServer        string       		`json:"from_server"`       
	Organization      string       		`json:"organization"`      
	Public            string       		`json:"public"`            
	RootVolume        RootVolume   		`json:"root_volume"`       
	State             string       		`json:"state"`             
	Project           string       		`json:"project"`           
	Tags              []string     		`json:"tags"`              
	Zone              string       		`json:"zone"`              
}

type Volume struct {
	ID               string        `json:"id"`               
	Name             string        `json:"name"`             
	ExportURI        string        `json:"export_uri"`       
	Size             int64         `json:"size"`             
	VolumeType       string        `json:"volume_type"`      
	CreationDate     string        `json:"creation_date"`    
	ModificationDate string        `json:"modification_date"`
	Organization     string        `json:"organization"`     
	Project          string        `json:"project"`          
	Tags             []string      `json:"tags,omitempty"`   
	Server           SecurityGroup `json:"server"`           
	State            string        `json:"state"`            
	Zone             string        `json:"zone"`             
	Boot             *string       `json:"boot,omitempty"`   
}

type SecurityGroup struct {
	ID   string `json:"id"`  
	Name string `json:"name"`
}

type RootVolume struct {
	ID         string `json:"id"`         
	Name       string `json:"name"`       
	Size       int64  `json:"size"`       
	VolumeType string `json:"volume_type"`
}

type Ipv6 struct {
	Address string `json:"address"`
	Gateway string `json:"gateway"`
	Netmask string `json:"netmask"`
}

type Location struct {
	ClusterID    string `json:"cluster_id"`   
	HypervisorID string `json:"hypervisor_id"`
	NodeID       string `json:"node_id"`      
	PlatformID   string `json:"platform_id"`  
	ZoneID       string `json:"zone_id"`      
}

type Maintenance struct {
}

type PlacementGroup struct {
	ID              string   `json:"id"`              
	Name            string   `json:"name"`            
	Organization    string   `json:"organization"`    
	Project         string   `json:"project"`         
	Tags            []string `json:"tags"`            
	PolicyMode      string   `json:"policy_mode"`     
	PolicyType      string   `json:"policy_type"`     
	PolicyRespected string   `json:"policy_respected"`
	Zone            string   `json:"zone"`            
}

type PrivateNIC struct {
	ID               string `json:"id"`                
	ServerID         string `json:"server_id"`         
	PrivateNetworkID string `json:"private_network_id"`
	MACAddress       string `json:"mac_address"`       
	State            string `json:"state"`             
}

type PublicIP struct {
	ID      string `json:"id"`     
	Address string `json:"address"`
	Dynamic string `json:"dynamic"`
}